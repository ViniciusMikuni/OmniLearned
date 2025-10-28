import typer

from omnilearned.train import run as run_training
from omnilearned.evaluate import run as run_evaluation
from omnilearned.dataloader import load_data
from omnilearned.train_hl import run as run_training_hl
from omnilearned.evaluate_hl import run as run_evaluation_hl

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
    resuming: bool = typer.Option(False, help="Resume training"),
    # Model Options
    num_feat: int = typer.Option(
        4,
        help="Number of input kinematic features (not considering PID or additional features)",
    ),
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
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
    # Model
    attn_drop: float = typer.Option(0.0, help="Dropout for attention layers"),
    mlp_drop: float = typer.Option(0.0, help="Dropout for mlp layers"),
    feature_drop: float = typer.Option(0.0, help="Dropout for input features"),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
):
    run_training(
        outdir,
        save_tag,
        pretrain_tag,
        dataset,
        path,
        wandb,
        fine_tune,
        resuming,
        num_feat,
        size,
        interaction,
        conditional,
        num_cond,
        use_pid,
        pid_idx,
        pid_dim,
        use_add,
        num_add,
        zero_add,
        use_clip,
        use_event_loss,
        num_classes,
        num_gen_classes,
        mode,
        batch,
        iterations,
        epoch,
        warmup_epoch,
        use_amp,
        optim,
        sched,
        b1,
        b2,
        lr,
        lr_factor,
        wd,
        attn_drop,
        mlp_drop,
        feature_drop,
        num_workers,
        clip_inputs=clip_inputs,
    )

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
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
):
    run_training_hl(
        outdir,
        save_tag,
        dataset,
        path,
        wandb,
        resuming,
        num_feat,
        conditional,
        num_cond,
        batch,
        iterations,
        epoch,
        warmup_epoch,
        optim,
        b1,
        b2,
        lr,
        wd,
        mlp_drop,
    )


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
    size: str = typer.Option("small", "--size", "-s", help="Model size"),
    interaction: bool = typer.Option(False, help="Use interaction matrix"),
    conditional: bool = typer.Option(False, help="Use global conditional features"),
    num_cond: int = typer.Option(3, help="Number of global conditioning features"),
    use_pid: bool = typer.Option(False, help="Use particle ID for training"),
    pid_idx: int = typer.Option(4, help="Index of the PID in the input array"),
    use_add: bool = typer.Option(
        False, help="Use additional features beyond kinematic information"
    ),
    num_add: int = typer.Option(4, help="Number of additional features"),
    use_event_loss: bool = typer.Option(
        False, help="Use additional classification loss between physics process"
    ),
    num_classes: int = typer.Option(
        2, help="Number of classes in the classification task"
    ),
    mode: str = typer.Option(
        "classifier", help="Task to run: classifier, generator, pretrain"
    ),
    # Training options
    batch: int = typer.Option(128, help="Batch size"),
    clip_inputs: bool = typer.Option(
        False, help="Clip input dataset to be within R=0.8 and atl least 500 MeV"
    ),
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
):
    run_evaluation(
        indir,
        outdir,
        save_tag,
        dataset,
        path,
        num_feat,
        size,
        interaction,
        conditional,
        num_cond,
        use_pid,
        pid_idx,
        use_add,
        num_add,
        use_event_loss,
        num_classes,
        mode,
        batch,
        num_workers,
        clip_inputs=clip_inputs,
    )



@app.command()
def evaluate_hl(
    # General Options
    indir: str = typer.Option(
        "", "--input_dir", "-i", help="Directory to input best model"
    ),
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
    num_workers: int = typer.Option(16, help="Number of workers for data loading"),
):
    run_evaluation_hl(
        indir,
        save_tag,
        dataset,
        path,
        num_feat,
        conditional,
        num_cond,
        batch,
        num_workers,
    )

    

@app.command()
def dataloader(
    dataset: str = typer.Option(
        "top", "--dataset", "-d", help="Dataset name to download"
    ),
    folder: str = typer.Option(
        "./", "--folder", "-f", help="Folder to save the dataset"
    ),
):
    for tag in ["train", "test", "val"]:
        print(tag)
        load_data(dataset, folder, dataset_type=tag, distributed=False)


if __name__ == "__main__":
    app()
