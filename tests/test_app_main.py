from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import __main__ as desktop


def test_desktop_launcher_connects_inference_config_to_shared_adapter():
    class_map = {17: "Almond"}
    config = SimpleNamespace(
        checkpoint="checkpoint.pth",
        class_map=class_map,
        variant="nano",
        device="cpu",
        confidence_floor=0.6,
    )
    detector = SimpleNamespace(model_label="rfdetr-nano:checkpoint@abc123")
    application = Mock()
    application.exec.return_value = 0
    with (
        patch.object(desktop, "load_inference_config", return_value=config) as load_config,
        patch.object(desktop, "load_detector", return_value=detector) as load_detector,
        patch.object(desktop, "QApplication", return_value=application),
        patch.object(desktop, "MainWindow") as main_window,
    ):
        assert desktop.main(["--inference-config", "inference.json"]) == 0

    load_config.assert_called_once()
    load_detector.assert_called_once_with(
        config.checkpoint, class_map, variant="nano", device="cpu"
    )
    main_window.assert_called_once_with(
        detector=detector,
        class_names=class_map,
        confidence_floor=0.6,
        model_label=detector.model_label,
    )
