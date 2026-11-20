"""Desktop launcher: python -m app."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.checkpoint import CheckpointError
from app.rfdetr_adapter import load_detector, load_inference_config
from app.ui.main_window import MainWindow


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="DINOS almond image analysis.")
    parser.add_argument(
        "--inference-config",
        type=Path,
        help="load an RF-DETR checkpoint and class map from this inference JSON",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    detector = None
    class_names = None
    confidence_floor = 0.25
    model_label = None
    if args.inference_config is not None:
        try:
            config = load_inference_config(args.inference_config)
            detector = load_detector(
                config.checkpoint, config.class_map,
                variant=config.variant, device=config.device,
            )
        except (CheckpointError, ImportError, OSError, ValueError) as error:
            parser.error(f"Could not load the RF-DETR model: {error}")
        class_names = config.class_map
        confidence_floor = config.confidence_floor
        model_label = detector.model_label

    application = QApplication([sys.argv[0]])
    window = MainWindow(
        detector=detector,
        class_names=class_names,
        confidence_floor=confidence_floor,
        model_label=model_label,
    )
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
