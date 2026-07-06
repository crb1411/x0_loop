from __future__ import annotations

from x0loop.training.generative_eval import _prune_fake_images_by_class


def test_prune_fake_images_by_class_keeps_one_per_label(tmp_path):
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()

    labels = [f"class{i}" for i in range(10)]
    for label in labels:
        for sample_idx in range(3):
            (fake_dir / f"sample_{sample_idx:06d}_y{label}_x0loop.png").write_bytes(b"png")

    kept, kept_labels = _prune_fake_images_by_class(str(fake_dir), per_class=1, max_classes=10)

    remaining = sorted(path.name for path in fake_dir.iterdir())
    assert kept == 10
    assert kept_labels == labels
    assert len(remaining) == 10
    for label in labels:
        assert sum(f"_y{label}_" in name for name in remaining) == 1


def test_prune_fake_images_by_class_keeps_unlabeled_fallback(tmp_path):
    fake_dir = tmp_path / "fake"
    fake_dir.mkdir()
    for sample_idx in range(20):
        (fake_dir / f"sample_{sample_idx:06d}_x0loop.png").write_bytes(b"png")

    kept, kept_labels = _prune_fake_images_by_class(str(fake_dir), per_class=1, max_classes=10)

    assert kept == 10
    assert kept_labels == []
    assert len(list(fake_dir.iterdir())) == 10
