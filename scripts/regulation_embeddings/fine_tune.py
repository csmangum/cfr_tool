"""Script for fine-tuning sentence transformer models on regulation data.

python -m scripts.regulation_embeddings.fine_tune --config config/config.yaml --data-dir data/agencies --output-dir models/fine_tuned --epochs 3 --batch-size 8 --learning-rate 2e-5
"""

import argparse
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from lxml import etree
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from .chunkers import XMLChunker
from .config import Config


def get_latest_regulation_files(data_dir: Path, target_date: str = None) -> List[Path]:
    """Get the most recent version of each regulation."""
    latest_files = {}  # (agency, title, chapter) -> (date, file_path)

    # Iterate through all XML files
    for xml_file in data_dir.rglob("*/xml/*.xml"):
        # Extract metadata from filename
        title_chapter_match = re.search(r"title_(\d+)_chapter_([^_]+)", xml_file.name)
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", xml_file.name)

        if not (title_chapter_match and date_match):
            continue

        title = title_chapter_match.group(1)
        chapter = title_chapter_match.group(2)
        date = date_match.group(1)
        agency = xml_file.parent.parent.name

        key = (agency, title, chapter)

        # If target_date is specified, only include files from that date
        if target_date and date != target_date:
            continue

        # Keep track of the latest date for each regulation
        if key not in latest_files or date > latest_files[key][0]:
            latest_files[key] = (date, xml_file)

    # Return only the file paths of the latest versions
    return [file_path for _, file_path in latest_files.values()]


def create_training_examples_from_xml(
    xml_files: List[Path], logger: logging.Logger = None
) -> List[InputExample]:
    """Create training examples from complete XML files."""
    # Store full text content of each file
    file_contents = {}

    for file_path in tqdm(xml_files, desc="Processing XML files"):
        try:
            # Parse XML and get full text content
            tree = etree.parse(str(file_path))
            root = tree.getroot()
            # Get all text content, joining with spaces
            text = " ".join(root.xpath("//text()"))
            # Clean up whitespace
            text = " ".join(text.split())

            if text.strip():  # Only store if there's actual content
                file_contents[file_path] = text

        except Exception as e:
            if logger:
                logger.warning(f"Error processing {file_path}: {str(e)}")
            continue

    if logger:
        logger.info(f"Processed {len(file_contents)} XML files")

    training_examples = []

    # Create training pairs from complete documents
    file_paths = list(file_contents.keys())
    for i, file_path in enumerate(file_paths):
        # Get documents from same agency as positives
        agency = file_path.parent.parent.name
        same_agency_files = [
            p for p in file_paths if p != file_path and p.parent.parent.name == agency
        ]

        # Get documents from different agencies as negatives
        different_agency_files = [
            p for p in file_paths if p.parent.parent.name != agency
        ]

        # Create positive pairs (same agency)
        if same_agency_files:
            positive_files = random.sample(
                same_agency_files, min(3, len(same_agency_files))
            )
            for pos_file in positive_files:
                training_examples.append(
                    InputExample(
                        texts=[file_contents[file_path], file_contents[pos_file]],
                        label=1.0,
                    )
                )

        # Create negative pairs (different agencies)
        if different_agency_files:
            negative_files = random.sample(
                different_agency_files, min(3, len(different_agency_files))
            )
            for neg_file in negative_files:
                training_examples.append(
                    InputExample(
                        texts=[file_contents[file_path], file_contents[neg_file]],
                        label=0.0,
                    )
                )

    if logger:
        total_positives = sum(1 for ex in training_examples if ex.label == 1.0)
        total_negatives = sum(1 for ex in training_examples if ex.label == 0.0)
        logger.info(f"Created {len(training_examples)} total training examples:")
        logger.info(f"  Positive pairs: {total_positives}")
        logger.info(f"  Negative pairs: {total_negatives}")

    return training_examples


class TrainingProgressCallback:
    """Callback to log training progress."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.best_loss = float("inf")
        self.steps_without_improvement = 0

    def __call__(self, score, epoch, steps):
        if score < self.best_loss:
            self.best_loss = score
            self.steps_without_improvement = 0
            self.logger.info(f"Epoch {epoch}, Step {steps}: New best loss: {score:.4f}")
        else:
            self.steps_without_improvement += 1
            self.logger.info(
                f"Epoch {epoch}, Step {steps}: Loss: {score:.4f} (no improvement for {self.steps_without_improvement} steps)"
            )


def collate_fn(batch):
    """Custom collate function to handle InputExample objects."""
    texts = []
    labels = []
    for example in batch:
        texts.extend(example.texts)
        labels.append(example.label)
    return {"texts": texts, "labels": torch.tensor(labels, dtype=torch.float)}


def fine_tune_model(
    config: Config,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
):
    """Fine-tune a sentence transformer model on regulation data."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Log training parameters
    logger.info("=== Training Configuration ===")
    logger.info(f"Base model: {config.embedder.model_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("===========================")

    # Load base model
    logger.info(f"Loading base model: {config.embedder.model_name}")
    model = SentenceTransformer(config.embedder.model_name)
    model = model.to(device)  # Move model to GPU/CPU
    logger.info(
        f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters"
    )

    try:
        # Get XML files
        xml_files = get_latest_regulation_files(data_dir)
        if not xml_files:
            raise ValueError(f"No XML files found in {data_dir}")

        logger.info(f"Found {len(xml_files)} XML files to process")

        # Create training examples from complete documents
        logger.info("Creating training examples from XML files")
        train_examples = create_training_examples_from_xml(xml_files, logger=logger)

        if not train_examples:
            raise ValueError("No training examples could be created")

        # Create data loader with custom collate function
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=batch_size, collate_fn=collate_fn
        )

        # Set up loss function
        train_loss = losses.CosineSimilarityLoss(model)

        # Set up optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Training loop
        logger.info("Starting fine-tuning")
        best_loss = float("inf")
        steps_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            batch_count = 0

            progress_bar = tqdm(
                train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True
            )

            for batch in progress_bar:
                optimizer.zero_grad()

                # Forward pass - modified to handle the new batch format
                texts = batch["texts"]
                labels = batch["labels"].to(device)

                # Process text pairs through the model to maintain gradients
                features = model.tokenize(texts)
                features = {k: v.to(device) for k, v in features.items()}
                embeddings = model(features)

                # Get sentence embeddings from the model output
                sentence_embeddings = embeddings["sentence_embedding"]

                # Split embeddings into pairs
                embeddings1 = sentence_embeddings[0::2]  # even indices
                embeddings2 = sentence_embeddings[1::2]  # odd indices

                # Calculate cosine similarity while maintaining gradients
                similarities = torch.nn.functional.cosine_similarity(
                    embeddings1, embeddings2
                )

                # Calculate loss
                loss_value = torch.nn.functional.mse_loss(similarities, labels)

                # Backward pass
                loss_value.backward()
                optimizer.step()

                # Update statistics
                total_loss += loss_value.item()
                batch_count += 1
                current_avg_loss = total_loss / batch_count

                # Update progress bar
                progress_bar.set_postfix(
                    {"loss": f"{current_avg_loss:.4f}", "best_loss": f"{best_loss:.4f}"}
                )

                # Log every 10 batches
                if batch_count % 10 == 0:
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_count}: Loss = {current_avg_loss:.4f}"
                    )

            # End of epoch logging
            epoch_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")

            # Save checkpoint if improved
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                steps_without_improvement = 0
                logger.info(f"New best loss: {best_loss:.4f}")
                model.save(str(output_dir / f"checkpoint-epoch-{epoch+1}"))
            else:
                steps_without_improvement += 1
                logger.info(f"No improvement for {steps_without_improvement} epochs")

        # Save final model
        model.save(str(output_dir / "final"))
        logger.info(
            f"Model fine-tuning complete. Final model saved to {output_dir / 'final'}"
        )

        # Log some example embeddings
        logger.info("Generating example embeddings...")
        sample_chunks = random.sample(train_examples, min(5, len(train_examples)))
        for example in sample_chunks:
            embedding = model.encode(example.texts[0])
            logger.info(f"Example embedding norm: {np.linalg.norm(embedding):.4f}")

    except Exception as e:
        logger.error(f"Error during fine-tuning: {str(e)}", exc_info=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a sentence transformer model on regulation data"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory containing agency data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning",
    )

    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run fine-tuning
    fine_tune_model(
        config=config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
