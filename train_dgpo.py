from collections.abc import Callable
import base64
from collections import deque
from datetime import datetime
import json
import pickle
from pathlib import Path
import random
import re
import subprocess
import tempfile
from typing import Any, Iterator, Optional
import zlib
import wandb
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel,
    GenerationConfig,
)
from loss import approx_kl_divergence, GRPOLoss, DGPOLoss
from replay_buffer import ReplayBuffer, Experience, join_experience_batch

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    Console = Group = Live = Panel = Table = None
    RICH_AVAILABLE = False


LBPP_RUST_PARQUET_URL = (
    "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/rust/test.parquet"
)


def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    bf16: bool = True,
    device_map=None,
    use_flash_attn: bool = False,  # set True if flash-attn installed
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer


def model_device(model: PreTrainedModel) -> torch.device:
    return next(model.parameters()).device


# DeepSeek Zero system prompt
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    reward_fn: Optional[Callable[[str], float]] = None,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:

    model.eval()

    # 1. format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": task,
        },
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model_device(model))

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(
        num_rollouts, 1
    )

    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    # 2. sample completions
    pad_token_id = tokenizer.pad_token_id
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_length,
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )

    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1] :] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    # 3. determine rewards
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        scorer = reward_fn or (lambda text: score_arithmetic_completion(text, oracle_answer))
        returns[i] = scorer(completion)

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions


def init_rng(seed: int) -> torch.Generator:
    random.seed(seed)
    return torch.manual_seed(seed)


def group_advantages(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (returns - returns.mean()) / (returns.std() + eps)


def sequence_log_probs_from_logits(
    logits: torch.tensor, output_ids: torch.tensor
) -> torch.Tensor:
    log_prob = F.log_softmax(logits, dim=-1)
    return log_prob.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(-1)


def sequences_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    device = model_device(model)
    sequence_ids = sequence_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"]
    log_probs = sequence_log_probs_from_logits(
        logits=logits[:, :-1].to(torch.float32),
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs


def sequences_log_probs_and_logits(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return both log probs (for chosen tokens) and full logits (for DGPO Hellinger)."""
    device = model_device(model)
    sequence_ids = sequence_ids.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(mask=(attention_mask == 0), value=1)
    output = model.forward(
        input_ids=sequence_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = output["logits"][:, :-1].to(torch.float32)  # [batch, seq-1, vocab]
    log_probs = sequence_log_probs_from_logits(
        logits=logits,
        output_ids=sequence_ids[:, 1:],
    )
    return log_probs, logits


def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def decode_lbpp_field(value: str) -> str | list | dict:
    """Decode LBPP's compressed code/test fields."""
    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(value.encode("utf-8")))))


def rust_prompt(row: dict) -> str:
    return f"""Write a Rust implementation for the following programming task.

Return only Rust code inside <answer>...</answer>. Do not include explanations.

Task:
{row["instruction"]}

Required function signature:
```rust
{row["signature"]}
```
"""


def read_lbpp_rust_prompts(max_rows: Optional[int] = None) -> list[dict]:
    dataset = load_dataset(
        "parquet",
        data_files=LBPP_RUST_PARQUET_URL,
        split="train",
    )
    rows = []
    for row in dataset:
        rows.append(
            {
                "task_id": row["task_id"],
                "question": rust_prompt(row),
                "answer": row["task_id"],
                "test_file": decode_lbpp_field(row["test_file"]),
            }
        )
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def extract_answer(completion: str) -> Optional[str]:
    answer_match = re.search(
        r"<answer>(.*?)</answer>",
        completion,
        flags=re.DOTALL,
    )
    if answer_match is None:
        return None
    return answer_match.group(1).strip()


def strip_markdown_code_fence(text: str) -> str:
    text = text.strip()
    fence_match = re.fullmatch(r"```(?:rust|rs)?\s*(.*?)```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def looks_like_rust_code(code: str) -> bool:
    return any(token in code for token in ("fn ", "pub fn ", "impl ", "use ", "let "))


def run_rust_command(
    code: str,
    test_file: str,
    command: list[str],
    timeout_seconds: int,
) -> Optional[subprocess.CompletedProcess]:
    with tempfile.TemporaryDirectory(prefix="tiny-dgpo-rust-") as tmp_dir:
        crate_dir = Path(tmp_dir)
        src_dir = crate_dir / "src"
        src_dir.mkdir()
        (crate_dir / "Cargo.toml").write_text(
            """[package]
name = "tiny_dgpo_rust_eval"
version = "0.1.0"
edition = "2021"

[dependencies]
approx = "0.5"
assert_fs = "1"
chrono = "0.4"
map-macro = "0.3"
serde_json = "1"
""",
            encoding="utf-8",
        )
        (src_dir / "code.rs").write_text(code, encoding="utf-8")
        (src_dir / "lib.rs").write_text(test_file, encoding="utf-8")

        try:
            return subprocess.run(
                command,
                cwd=crate_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None


def run_rust_syntax_check(code: str, timeout_seconds: int) -> bool:
    with tempfile.TemporaryDirectory(prefix="tiny-dgpo-rust-syntax-") as tmp_dir:
        code_path = Path(tmp_dir) / "code.rs"
        output_path = Path(tmp_dir) / "libcode.rlib"
        code_path.write_text(code, encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    "rustc",
                    "--edition=2021",
                    "--crate-type",
                    "lib",
                    str(code_path),
                    "-o",
                    str(output_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return False
    return result.returncode == 0


def score_arithmetic_completion(completion: str, oracle_answer: str) -> float:
    answer = extract_answer(completion)
    if answer is None:
        return 0.0
    answer = answer.strip()
    if answer == oracle_answer:
        return 1.0
    if oracle_answer in answer:
        return 0.5
    return 0.01


def score_rust_completion(
    completion: str,
    test_file: str,
    timeout_seconds: int = 20,
) -> float:
    answer = extract_answer(completion)
    if answer is None:
        return 0.0

    code = strip_markdown_code_fence(answer)
    if not code:
        return 0.0
    if not looks_like_rust_code(code):
        return 0.02

    if not run_rust_syntax_check(code, timeout_seconds):
        return 0.05

    check = run_rust_command(
        code,
        test_file,
        ["cargo", "check", "--quiet"],
        timeout_seconds,
    )
    if check is None or check.returncode != 0:
        return 0.10

    test = run_rust_command(
        code,
        test_file,
        ["cargo", "test", "--quiet"],
        timeout_seconds,
    )
    if test is None:
        return 0.25
    return 1.0 if test.returncode == 0 else 0.25


def rust_completion_stats(completions: list[str]) -> dict[str, int]:
    tagged = 0
    code_like = 0
    syntax_valid = 0
    for completion in completions:
        answer = extract_answer(completion)
        if answer is None:
            continue
        tagged += 1
        code = strip_markdown_code_fence(answer)
        if looks_like_rust_code(code):
            code_like += 1
        if run_rust_syntax_check(code, timeout_seconds=10):
            syntax_valid += 1
    return {"tagged": tagged, "code_like": code_like, "syntax_valid": syntax_valid}


class JsonlMetricLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self.file = self.path.open("a", encoding="utf-8")

    def log(self, event: str, **fields) -> None:
        row = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        self.file.write(json.dumps(row, sort_keys=True) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "JsonlMetricLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TrainingDashboard:
    def __init__(self, enabled: bool = True, max_events: int = 10) -> None:
        self.enabled = enabled and RICH_AVAILABLE
        self.console = Console() if self.enabled else None
        self.live = None
        self.events = deque(maxlen=max_events)
        self.state = {
            "step": 0,
            "task_source": "",
            "model_name": "",
            "policy_device": "",
            "reference_device": "",
            "num_prompts": 0,
            "run_dir": "",
            "last_return": 0.0,
            "total_return": 0.0,
            "rollouts": 0,
            "updates": 0,
            "skips": 0,
            "last_dgpo_score": 0.0,
            "last_grad_norm": 0.0,
            "last_loss": 0.0,
        }

    def __enter__(self) -> "TrainingDashboard":
        if self.enabled:
            self.live = Live(
                self.render(),
                console=self.console,
                refresh_per_second=4,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.live.start(refresh=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.live is not None:
            self.live.stop()

    def update(self, **kwargs) -> None:
        self.state.update(kwargs)
        if self.live is not None:
            self.live.update(self.render(), refresh=True)

    def log(self, message: str) -> None:
        if self.enabled:
            self.events.appendleft(message)
            self.update()
        else:
            print(message)

    def record_rollout(
        self,
        task_label: str,
        reward_sum: float,
        replay_buffer_size: int,
        sequence_shape: tuple[int, ...],
        tagged: Optional[int] = None,
        code_like: Optional[int] = None,
        syntax_valid: Optional[int] = None,
    ) -> None:
        self.state["rollouts"] += 1
        tag_text = ""
        if tagged is not None and code_like is not None and syntax_valid is not None:
            tag_text = (
                f" tagged={tagged}/{sequence_shape[0]}"
                f" code={code_like}/{sequence_shape[0]}"
                f" syntax={syntax_valid}/{sequence_shape[0]}"
            )
        self.log(
            f"{task_label} reward={reward_sum:.2f} replay={replay_buffer_size} "
            f"shape={sequence_shape}{tag_text}"
        )

    def record_step_return(self, step: int, value: float) -> None:
        self.state["step"] = step
        self.state["last_return"] = value
        self.state["total_return"] += value
        self.update()

    def record_skip(self, step: int) -> None:
        self.state["skips"] += 1
        self.log(f"step {step}: no relative reward signal, skipped optimization")

    def record_update(self, step: int, loss: float, dgpo_score: float, grad_norm: float) -> None:
        self.state["step"] = step
        self.state["updates"] += 1
        self.state["last_loss"] = loss
        self.state["last_dgpo_score"] = dgpo_score
        self.state["last_grad_norm"] = grad_norm
        self.log(
            f"step {step}: update loss={loss:.4f} dgpo={dgpo_score:.4f} grad={grad_norm:.4f}"
        )

    def render(self):
        status = Table.grid(expand=True)
        status.add_column(ratio=1)
        status.add_column(ratio=1)
        status.add_row(
            f"[bold]step[/bold] {self.state['step']}    "
            f"[bold]source[/bold] {self.state['task_source']}    "
            f"[bold]prompts[/bold] {self.state['num_prompts']}",
            f"[bold]model[/bold] {self.state['model_name']}",
        )
        status.add_row(
            f"[bold]policy[/bold] {self.state['policy_device']}    "
            f"[bold]reference[/bold] {self.state['reference_device']}",
            f"[bold]rollouts[/bold] {self.state['rollouts']}    "
            f"[bold]updates[/bold] {self.state['updates']}    "
            f"[bold]skips[/bold] {self.state['skips']}",
        )
        status.add_row(
            f"[bold]last return[/bold] {self.state['last_return']:.3f}    "
            f"[bold]total return[/bold] {self.state['total_return']:.3f}",
            f"[bold]loss[/bold] {self.state['last_loss']:.4f}    "
            f"[bold]dgpo[/bold] {self.state['last_dgpo_score']:.4f}    "
            f"[bold]grad[/bold] {self.state['last_grad_norm']:.4f}",
        )
        status.add_row(
            f"[bold]run dir[/bold] {self.state['run_dir']}",
            "",
        )

        events = Table(title="Recent Rollouts / Updates", expand=True)
        events.add_column("event", overflow="fold")
        for event in self.events:
            events.add_row(event)
        if not self.events:
            events.add_row("waiting for first rollout...")

        return Panel(
            Group(status, events),
            title="tiny-dgpo",
            border_style="cyan",
        )


def main():
    seed = 42
    wandb_project = None  # "tiny_dgpo"
    device_index = 0
    # 24GB GPU profile. DGPO computes full-vocab logits for both policy and
    # reference models, so keep batches small when using 2B+ models.
    model_name = "Qwen/Qwen3.5-2B"
    checkpoint_path = Path("./output_dgpo")
    checkpoint_interval = 20
    train_batch_size = 1  # conservative for DGPO logits storage
    lr = 1e-6           # paper uses 1e-6
    weight_decay = 0.1  # paper uses 0.1
    clip_eps = 0.2
    task_source = "lbpp_rust"  # "lbpp_rust" or "arithmetic"
    use_tui = True
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")

    # DGPO hyperparameters (from paper Table 4 & 5)
    dgpo_tau = 0.5      # temperature for softmax reweighting (optimal per paper)
    dgpo_kappa = 1.0    # entropy gating exponent (optimal per paper)

    group_size = 4      # paper uses G=16, reduced for memory
    rollouts_per_step = 2
    epochs_per_step = 1
    max_norm = 1.0  # gradient clipping

    # rollout params
    max_length = 128    # full-vocab DGPO logits make long code rollouts expensive
    top_p = 1.0
    temperature = 1.0

    policy_device = torch.device("cuda", device_index)
    reference_device = torch.device("cuda", 1) if torch.cuda.device_count() > 1 else policy_device
    device = policy_device
    cpu_device = torch.device("cpu")
    init_rng(seed)

    print(f"policy_device={policy_device}, reference_device={reference_device}")
    reference_model, _ = load_model(model_name, device_map={"": str(reference_device)})
    model, tokenizer = load_model(model_name, device_map={"": str(policy_device)})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        foreach=False,
    )

    reference_model.eval()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    pad_token_id = tokenizer.pad_token_id

    if task_source == "lbpp_rust":
        prompts = read_lbpp_rust_prompts(max_rows=149)
    elif task_source == "arithmetic":
        prompts = read_prompts(
            "data/math_tasks.jsonl",
            predicate=lambda x: len(x["question"]) < 128
            and x["num_terms"] <= 3
            and x["num_digits"] <= 3,
            max_rows=64 * 1024,
        )
    else:
        raise ValueError(f"Unknown task_source={task_source}")
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = DGPOLoss(clip_eps=clip_eps, tau=dgpo_tau, kappa=dgpo_kappa)

    if wandb_project is None:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=wandb_project)

    dashboard = TrainingDashboard(enabled=use_tui)
    dashboard.update(
        task_source=task_source,
        model_name=model_name,
        policy_device=str(policy_device),
        reference_device=str(reference_device),
        num_prompts=len(prompts),
        run_dir=str(run_dir),
    )

    with JsonlMetricLogger(run_dir) as metric_logger, dashboard:
      metric_logger.log(
          "config",
          model_name=model_name,
          task_source=task_source,
          group_size=group_size,
          rollouts_per_step=rollouts_per_step,
          train_batch_size=train_batch_size,
          max_new_tokens=max_length,
          tau=dgpo_tau,
          kappa=dgpo_kappa,
          lr=lr,
          weight_decay=weight_decay,
          clip_eps=clip_eps,
          policy_device=str(policy_device),
          reference_device=str(reference_device),
      )
      for k, prompt_batch in enumerate(prompt_loader):
        rollout_returns = []

        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]
        test_files = prompt_batch.get("test_file")

        with torch.no_grad():
            for idx, (q, a) in enumerate(zip(questions, answers)):
                reward_fn = None
                if task_source == "lbpp_rust":
                    test_file = test_files[idx]
                    reward_fn = lambda completion, test_file=test_file: score_rust_completion(
                        completion,
                        test_file,
                    )
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=group_size,
                    reward_fn=reward_fn,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                )

                task_label = a if task_source == "lbpp_rust" else q
                tagged = None
                code_like = None
                syntax_valid = None
                if task_source == "lbpp_rust":
                    stats = rust_completion_stats(completions)
                    tagged = stats["tagged"]
                    code_like = stats["code_like"]
                    syntax_valid = stats["syntax_valid"]
                reward_values = [
                    float(value)
                    for value in returns.squeeze(-1).detach().cpu().tolist()
                ]
                metric_logger.log(
                    "rollout",
                    step=k,
                    task=task_label,
                    reward_sum=float(returns.sum().item()),
                    rewards=reward_values,
                    tagged=tagged,
                    code_like=code_like,
                    syntax_valid=syntax_valid,
                    sequence_shape=list(sequence_ids.shape),
                    replay_buffer_size=len(replay_buffer),
                )
                dashboard.record_rollout(
                    task_label=task_label,
                    reward_sum=returns.sum().item(),
                    replay_buffer_size=len(replay_buffer),
                    sequence_shape=tuple(sequence_ids.shape),
                    tagged=tagged,
                    code_like=code_like,
                    syntax_valid=syntax_valid,
                )
                rollout_returns.append(returns.cpu())

                advantages = group_advantages(returns)
                if advantages.abs().sum().item() == 0:
                    dashboard.log(f"{task_label}: no group-relative signal, skipped logits")
                    metric_logger.log(
                        "skip_logits",
                        step=k,
                        task=task_label,
                        reason="no_group_relative_signal",
                    )
                    continue
                attention_mask = sequence_ids != pad_token_id

                # Get log probs from policy (for importance sampling ratio)
                log_probs = sequences_log_probs(
                    model=model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )
                # DGPO: get ref logits for Hellinger distance computation
                log_probs_ref, ref_logits = sequences_log_probs_and_logits(
                    model=reference_model,
                    sequence_ids=sequence_ids,
                    attention_mask=attention_mask,
                )

                experience = Experience(
                    sequences=sequence_ids,
                    action_log_probs=log_probs,
                    log_probs_ref=log_probs_ref,
                    returns=returns,
                    advantages=advantages,
                    attention_mask=attention_mask,
                    action_mask=action_mask,
                    kl=None,
                    policy_logits=None,  # not needed - we compute fresh during training
                    ref_logits=ref_logits,
                )
                replay_buffer.append(experience.to(cpu_device))

        torch.cuda.empty_cache()
        episode_return_sum = torch.stack(rollout_returns).sum()
        dashboard.record_step_return(k, episode_return_sum.item())
        metric_logger.log(
            "step_return",
            step=k,
            return_sum=float(episode_return_sum.item()),
        )
        wandb.log({"returns": episode_return_sum})

        advantage_signal = sum(
            item.advantages.abs().sum().item()
            for item in replay_buffer.items
            if item.advantages is not None
        )
        if advantage_signal == 0:
            dashboard.record_skip(k)
            metric_logger.log(
                "skip_optimization",
                step=k,
                reason="no_relative_reward_signal",
            )
            continue

        experience_sampler = DataLoader(
            replay_buffer,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=join_experience_batch,
        )

        for step_epoch in range(epochs_per_step):
            model.train()

            for exp in experience_sampler:
                exp: Experience

                exp = exp.to(device)

                optimizer.zero_grad()

                # DGPO: need current logits for Hellinger computation during training
                log_probs, current_policy_logits = sequences_log_probs_and_logits(
                    model, sequence_ids=exp.sequences, attention_mask=exp.attention_mask
                )

                loss, train_metrics = objective(
                    log_probs=log_probs,
                    experience=exp,
                    policy_logits=current_policy_logits,
                    ref_logits=exp.ref_logits,
                )

                if not loss.isfinite():
                    dashboard.log(
                        f"step {k}: non-finite loss={loss}, advantages={exp.advantages}"
                    )
                    metric_logger.log(
                        "skip_update",
                        step=k,
                        reason="non_finite_loss",
                        loss=str(loss),
                    )
                    continue

                loss.backward()
                grad_norm = clip_grad_norm_(model.parameters(), max_norm=max_norm)
                if not torch.isfinite(grad_norm):
                    dashboard.log(
                        f"step {k}: non-finite grad norm={grad_norm}, skipped optimizer step"
                    )
                    metric_logger.log(
                        "skip_update",
                        step=k,
                        reason="non_finite_grad_norm",
                        grad_norm=str(grad_norm),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                dashboard.record_update(
                    step=k,
                    loss=loss.item(),
                    dgpo_score=train_metrics["dgpo_score_mean"],
                    grad_norm=grad_norm.item(),
                )
                metric_logger.log(
                    "update",
                    step=k,
                    epoch=step_epoch,
                    loss=float(loss.item()),
                    dgpo_score=float(train_metrics["dgpo_score_mean"]),
                    grad_norm=float(grad_norm.item()),
                )
                wandb.log({"dgpo_score": train_metrics['dgpo_score_mean'], "grad_norm": grad_norm})

                optimizer.step()

        if (
            checkpoint_path is not None
            and checkpoint_interval is not None
            and (k + 1) % checkpoint_interval == 0
        ):
            model.save_pretrained(checkpoint_path / f"step_{k}")

    if checkpoint_path is not None:
        model.save_pretrained(checkpoint_path / f"step_{k}")


if __name__ == "__main__":
    main()
