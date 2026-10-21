import typer

from omnilearned.config import resolve
from omnilearned.evaluate import run as run_evaluation
from omnilearned.evaluate_hl import run as run_evaluation_hl
from omnilearned.train import run as run_training
from omnilearned.train_hl import run as run_training_hl

app = typer.Typer(
    help="OmniLearned: A unified deep learning approach for particle physics",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command()
def train(
    # General Options
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output best model"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    pretrain_tag: str = typer.Option(
        "", help="Tag given to pretrained checkpoint model"
    ),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    wandb: bool = typer.Option(False, help="use wandb logging"),
    fine_tune: bool = typer.Option(False, help="Fine tune the model"),
    freeze: bool = typer.Option(
        False, help="Freeze body model weights during fine-tuning"
    ),
    resuming: bool = typer.Option(False, help="Resume training"),
    pretrain_path: str = typer.Option(
        "", help="Explicit local pretrained checkpoint for fine-tuning"
    ),
    # Model Options
    num_feat: int = typer.Option(
        4,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    num_part: int = typer.Option(
        150,
        help="Number of Particles in the Point Cloud. Only necesssary for encoder training",
    ),
    num_tokens: int = typer.Option(
        4, "--num_tokens", "--num-tokens", min=1,
        help="Number of learned tokens, also the encoder latent token count"
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    local_interaction: bool = typer.Option(False, help="Use local interaction matrix"),
    num_coord: int = typer.Option(
        2, help="Number of features for distance calculation"
    ),
    K: int = typer.Option(10, "--k", "--K", help="Number of k-neighbors"),
    interaction_type: str = typer.Option("lhc", help="Type of interaction"),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    use_pid: bool = typer.Option(False, help="Use particle ID for training"),
    pid_idx: int = typer.Option(4, help="Index of the PID in the input array"),
    pid_dim: int = typer.Option(9, help="Number of unique PIDs"),
    use_add: bool = typer.Option(
        False, help="Use additional features beyond kinematic information"
    ),
    num_add: int = typer.Option(4, help="Number of additional features"),
    zero_add: bool = typer.Option(
        False,
        help="Load the model with additional blocks but zero the inputs from the dataloader",
    ),
    use_clip: bool = typer.Option(False, help="Use CLIP loss during training"),
    ot_iterations: int = typer.Option(
        20, "--ot-iterations", "--ot_iterations", min=1,
        help="Sinkhorn iterations for each decoder reconstruction loss",
    ),
    use_event_loss: bool = typer.Option(
        False, help="Use additional classification loss between physics process"
    ),
    num_classes: int = typer.Option(
        2, help="Number of classes in the classification task"
    ),
    num_gen_classes: int = typer.Option(
        1, help="Number of classes in the particle classification task"
    ),
    mode: str = typer.Option(
        "classifier", help="Task to run: classifier, generator, pretrain"
    ),
    # Training options
    batch: int = typer.Option(64, help="Batch size"),
    iterations: int = typer.Option(-1, help="Number of iterations per pass"),
    epoch: int = typer.Option(10, help="Number of epochs"),
    warmup_epoch: int = typer.Option(0, help="Number of learning rate warmup epochs"),
    use_amp: bool = typer.Option(False, help="Use amp"),
    clip_inputs: bool = typer.Option(
        False, help="Clip input dataset to be within R=0.8 and atl least 500 MeV"
    ),
    # Optimizer
    optim: str = typer.Option("lion", help="optimizer to use"),
    sched: str = typer.Option("cosine", help="lr scheduler to use"),
    b1: float = typer.Option(0.95, help="Lion b1"),
    b2: float = typer.Option(0.98, help="Lion b2"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    lr_factor: float = typer.Option(
        1.0, help="Learning rate factor for new layers during fine-tuning"
    ),
    wd: float = typer.Option(0.0, help="Weight decay"),
    nevts: int = typer.Option(-1, help="Maximum number of events to use"),
    # Model
    attn_drop: float = typer.Option(0.0, help="Dropout for attention layers"),
    mlp_drop: float = typer.Option(0.0, help="Dropout for mlp layers"),
    feature_drop: float = typer.Option(0.0, help="Dropout for input features"),
    num_workers: int = typer.Option(2, help="Number of workers for data loading"),
    config: str = typer.Option("", help="YAML overrides matching CLI flags"),
    seed: int = typer.Option(42),
    check_remote: bool = typer.Option(True),
    base_url: str = typer.Option("https://portal.nersc.gov/cfs/dasrepo/omnilearned/"),
    persistent_workers: bool = typer.Option(True),
    prefetch_factor: int = typer.Option(2),
    max_open_files: int = typer.Option(64),
    qk_norm: bool = typer.Option(False),
    attention_backend: str = typer.Option("auto"),
    precision: str = typer.Option("fp32"),
    val_iterations: int = typer.Option(-1),
    grad_accumulation: int = typer.Option(1),
    grad_clip: float = typer.Option(1.0),
    ema: bool = typer.Option(True),
    ema_decay: float = typer.Option(0.999),
    compile_model: bool = typer.Option(False),
    activation_checkpointing: bool = typer.Option(False),
    bucket_size: int = typer.Option(0),
):
    execute(run_training, locals())


@app.command()
def train_hl(
    # General Options
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output best model"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    wandb: bool = typer.Option(False, help="use wandb logging"),
    resuming: bool = typer.Option(False, help="Resume training"),
    # Model Options
    num_feat: int = typer.Option(
        3,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(1, help="Number of global conditioning features"),
    # Training options
    batch: int = typer.Option(64, help="Batch size"),
    iterations: int = typer.Option(-1, help="Number of iterations per pass"),
    epoch: int = typer.Option(10, help="Number of epochs"),
    warmup_epoch: int = typer.Option(0, help="Number of learning rate warmup epochs"),
    # Optimizer
    optim: str = typer.Option("lion", help="optimizer to use"),
    b1: float = typer.Option(0.95, help="Lion b1"),
    b2: float = typer.Option(0.98, help="Lion b2"),
    lr: float = typer.Option(5e-5, help="Learning rate"),
    wd: float = typer.Option(0.0, help="Weight decay"),
    # Model
    mlp_drop: float = typer.Option(0.0, help="Dropout for mlp layers"),
    num_workers: int = typer.Option(0, help="Number of workers for data loading"),
    config: str = typer.Option("", help="YAML overrides matching CLI flags"),
    seed: int = typer.Option(42),
    check_remote: bool = typer.Option(True),
    base_url: str = typer.Option("https://portal.nersc.gov/cfs/dasrepo/omnilearned/"),
    persistent_workers: bool = typer.Option(False),
    prefetch_factor: int = typer.Option(1),
    max_open_files: int = typer.Option(8),
    precision: str = typer.Option("fp32"),
    val_iterations: int = typer.Option(-1),
    grad_accumulation: int = typer.Option(1),
    grad_clip: float = typer.Option(1.0),
    ema: bool = typer.Option(True),
    ema_decay: float = typer.Option(0.999),
    compile_model: bool = typer.Option(False),
    activation_checkpointing: bool = typer.Option(False),
    bucket_size: int = typer.Option(0),
):
    execute(run_training_hl, locals())


@app.command()
def evaluate(
    # General Options
    indir: str = typer.Option(
        "", "--input_dir", "-i", help="Directory to input best model"
    ),
    outdir: str = typer.Option(
        "", "--output_dir", "-o", help="Directory to output evaluation results"
    ),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    # Model Options
    num_feat: int = typer.Option(
        4,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    num_part: int = typer.Option(
        150,
        help="Number of Particles in the Point Cloud. Only necesssary for encoder training",
    ),
    num_tokens: int = typer.Option(
        4, "--num_tokens", "--num-tokens", min=1,
        help="Number of learned tokens, also the encoder latent token count"
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    local_interaction: bool = typer.Option(False, help="Use local interaction matrix"),
    num_coord: int = typer.Option(
        2, help="Number of features for distance calculation"
    ),
    K: int = typer.Option(10, "--k", "--K", help="Number of k-neighbors"),
    interaction_type: str = typer.Option("lhc", help="Type of interaction"),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    use_pid: bool = typer.Option(False, help="Use particle ID for training"),
    pid_idx: int = typer.Option(4, help="Index of the PID in the input array"),
    pid_dim: int = typer.Option(9, help="Number of unique PIDs"),
    use_add: bool = typer.Option(
        False, help="Use additional features beyond kinematic information"
    ),
    num_add: int = typer.Option(4, help="Number of additional features"),
    zero_add: bool = typer.Option(
        False, help="Zero additional features while retaining the feature block"
    ),
    use_event_loss: bool = typer.Option(
        False, help="Use additional classification loss between physics process"
    ),
    num_classes: int = typer.Option(
        2, help="Number of classes in the classification task"
    ),
    num_gen_classes: int = typer.Option(
        1, help="Number of classes in the particle segmentation task"
    ),
    mode: str = typer.Option(
        "classifier", help="Task to run: classifier, regression, segmentation, generator, encoder, decoder"
    ),
    sbi: bool = typer.Option(
        False,
        help="Run SBI parameter inference. Currently only supported for --mode generator",
    ),
    # Training options
    batch: int = typer.Option(128, help="Batch size"),
    clip_inputs: bool = typer.Option(
        False, help="Clip input dataset to be within R=0.8 and atl least 500 MeV"
    ),
    num_workers: int = typer.Option(2, help="Number of workers for data loading"),
    config: str = typer.Option("", help="YAML overrides matching CLI flags"),
    seed: int = typer.Option(42),
    check_remote: bool = typer.Option(True),
    base_url: str = typer.Option("https://portal.nersc.gov/cfs/dasrepo/omnilearned/"),
    persistent_workers: bool = typer.Option(False),
    prefetch_factor: int = typer.Option(2),
    max_open_files: int = typer.Option(128),
    qk_norm: bool = typer.Option(False),
    attention_backend: str = typer.Option("auto"),
    nsteps: int = typer.Option(256),
    use_ema: bool = typer.Option(True),
    checkpoint_path: str = typer.Option(""),
    scan_ranges: str = typer.Option("[[6.5, 7.3], [4.5, 5.5]]"),
    scan_points: int = typer.Option(7),
    scan_steps: int = typer.Option(32),
    scan_refinements: int = typer.Option(2),
    scan_probes: int = typer.Option(1),
    scan_candidate_batch: int = typer.Option(4),
):
    execute(run_evaluation, locals())


@app.command()
def evaluate_hl(
    # General Options
    indir: str = typer.Option(
        "", "--input_dir", "-i", help="Directory to input best model"
    ),
    outdir: str = typer.Option("", "--output_dir", "-o", help="Output saved files"),
    save_tag: str = typer.Option("", help="Extra tag for checkpoint model"),
    dataset: str = typer.Option("top", help="Dataset to load"),
    path: str = typer.Option("/pscratch/sd/v/vmikuni/datasets", help="Dataset path"),
    # Model Options
    num_feat: int = typer.Option(
        3,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    # Training options
    batch: int = typer.Option(128, help="Batch size"),
    num_workers: int = typer.Option(0, help="Number of workers for data loading"),
    config: str = typer.Option("", help="YAML overrides matching CLI flags"),
    seed: int = typer.Option(42),
    check_remote: bool = typer.Option(True),
    base_url: str = typer.Option("https://portal.nersc.gov/cfs/dasrepo/omnilearned/"),
    persistent_workers: bool = typer.Option(False),
    prefetch_factor: int = typer.Option(1),
    max_open_files: int = typer.Option(8),
    nsteps: int = typer.Option(256),
    use_ema: bool = typer.Option(True),
    checkpoint_path: str = typer.Option(""),
    condition_file: str = typer.Option("outputs_mass_bsm.npz"),
    max_part: int = typer.Option(100),
):
    execute(run_evaluation_hl, locals())


@app.command()
def dataloader(
    dataset: str = typer.Option(
        "top", "--dataset", "-d", help="Dataset name to download"
    ),
    folder: str = typer.Option(
        "./", "--folder", "-f", help="Folder to save the dataset"
    ),
    config: str = typer.Option("", help="YAML overrides matching CLI flags"),
    seed: int = typer.Option(42),
    check_remote: bool = typer.Option(True),
    base_url: str = typer.Option("https://portal.nersc.gov/cfs/dasrepo/omnilearned/"),
    persistent_workers: bool = typer.Option(False),
    prefetch_factor: int = typer.Option(1),
    max_open_files: int = typer.Option(8),
):
    options = resolve(locals())
    import torch.distributed as dist

    from omnilearned.dataloader import prepare_dataset
    from omnilearned.utils import ddp_setup

    ddp_setup()
    try:
        for split in ("train", "val", "test"):
            prepare_dataset(
                options["dataset"],
                options["folder"],
                split,
                options["base_url"],
                options["check_remote"],
            )
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def execute(function, options):
    try:
        return function(**resolve(options))
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


@app.command(
    "submit",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def submit_command(
    ctx: typer.Context,
    account: str = typer.Option("m3246"),
    nodes: int = typer.Option(16),
    gpus_per_node: int = typer.Option(4),
    cpus_per_task: int = typer.Option(32),
    qos: str = typer.Option("regular"),
    time: str = typer.Option("04:00:00"),
    job_name: str = typer.Option("omnilearned"),
    output: str = typer.Option("job_log_%j.out"),
    constraint: str = typer.Option("gpu"),
    module: str = typer.Option("pytorch"),
    workdir: str = typer.Option(""),
    master_port: int = typer.Option(29500),
    script: str = typer.Option(
        "", help="New script filename; default is unique under jobs/"
    ),
    dry_run: bool = typer.Option(False),
):
    """Submit to Slurm. Put training/evaluation flags after -- (e.g. -- train --dataset top)."""
    from datetime import datetime
    from uuid import uuid4

    from omnilearned.submit import render, submit, validated_args

    try:
        arguments = list(ctx.args)
        if not arguments:
            raise ValueError(
                "Provide a command after --, for example -- train --dataset top"
            )
        arguments = validated_args(arguments)
        contents = render(
            arguments,
            account=account,
            nodes=nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_task=cpus_per_task,
            qos=qos,
            time=time,
            job_name=job_name,
            output=output,
            constraint=constraint,
            module=module,
            workdir=workdir or None,
            master_port=master_port,
        )
        filename = (
            script
            or f"jobs/{job_name}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.slurm"
        )
        submit(filename, contents, dry_run=dry_run)
    except (ValueError, RuntimeError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
