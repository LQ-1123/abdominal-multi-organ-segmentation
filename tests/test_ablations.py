import io
from zipfile import ZipFile

import numpy as np
import pytest
import torch
from scipy import ndimage as ndi

from scripts.run_ablations import STRUCTURE, archive_pairs, measure, morphology, sample


def test_archive_requires_complete_pairs():
    with ZipFile(io.BytesIO(), "w") as archive:
        archive.writestr("fold0_val/amos_0005.img.npy", b"")
        with pytest.raises(ValueError, match="配对"):
            archive_pairs(archive)
        archive.writestr("fold0_val/amos_0005.lbl.npy", b"")
        assert list(archive_pairs(archive)) == ["amos_0005"]


def test_even_sample_is_fixed_and_keeps_both_ends():
    ids = [str(i) for i in range(10)]
    assert sample(ids, 4) == ["0", "3", "6", "9"]


def test_cropped_morphology_matches_whole_volume_for_single_class():
    label = np.zeros((24, 24, 24), dtype=np.uint8)
    label[:13, 4:17, 6:19] = 1
    label[6, 12, 12] = 0
    label[1, 2, 3] = 1
    want = ndi.binary_closing(ndi.binary_opening(label, structure=STRUCTURE), structure=STRUCTURE)
    got = morphology(label)
    np.testing.assert_array_equal(got, want)
    assert got[6, 12, 12] == 1
    assert got[1, 2, 3] == 0


def test_morphology_does_not_claim_other_organ_voxels():
    label = np.zeros((24, 24, 24), dtype=np.uint8)
    label[3:19, 3:19, 3:19] = 1
    label[7:14, 7:14, 7:14] = 2
    got = morphology(label)
    assert got[10, 10, 10] == 2
    assert not (got[label == 2] == 1).any()


def test_metrics_treat_absent_truth_and_predictions_differently():
    truth = np.zeros((12, 12, 12), dtype=np.uint8)
    pred = truth.copy()
    truth[2:6, 2:6, 2:6] = 1
    pred[3:7, 2:6, 2:6] = 1
    truth[10, 10, 10] = 2
    pred[0, 0, 0] = 3
    a, b, c, *_ = measure(pred, truth, surfaces=True)
    assert a["dice"] == 0.75 and a["hd95_mm"] == 1.5 and a["components"] == 1
    assert b["dice"] == 0 and b["hd95_mm"] is None and b["pred_missing"] == 1
    assert c["dice"] is None and c["components"] == 1 and c["pred_missing"] == 0


def test_selective_morphology_leaves_unlisted_organs_untouched():
    label = np.zeros((24, 24, 24), dtype=np.uint8)
    label[3:12, 3:12, 3:12] = 1
    label[1, 1, 1] = 1          # 开运算会删掉的孤立点
    label[14:22, 14:22, 14:22] = 2
    label[0, 23, 23] = 2
    got = morphology(label, organs={1})
    assert got[1, 1, 1] == 0
    np.testing.assert_array_equal(got == 2, label == 2)


def test_butterworth_keeps_constant_and_damps_checkerboard():
    from src.preprocess.classic import butterworth_lowpass

    flat = np.full((16, 16, 16), 40, dtype=np.int16)
    np.testing.assert_allclose(butterworth_lowpass(flat), 40, atol=1e-3)
    z, y, x = np.indices((16, 16, 16))
    checker = (((z + y + x) % 2) * 200 - 100).astype(np.int16)
    assert np.abs(butterworth_lowpass(checker)).max() < 10


def test_classic_methods_return_model_range_and_cache_roundtrip():
    from src.preprocess import classic

    rng = np.random.default_rng(0)
    hu = rng.normal(40, 60, size=(12, 70, 70)).astype(np.int16)
    for method in classic.METHODS:
        x = classic.apply(hu, method)
        assert x.dtype == np.float32 and x.shape == hu.shape
        assert x.min() >= 0 and x.max() <= 1
        back = classic.window(classic.to_cache_hu(x))
        assert np.abs(back - x).max() <= 1 / 800 + 1e-6
    np.testing.assert_allclose(classic.apply(hu, "none"), (np.clip(hu, -150, 250) + 150) / 400, atol=1e-6)


def test_dataset_patch_shapes_and_fold_split(tmp_path):
    from src.data.amos_dataset import AMOSDataset, make_folds

    img = np.full((40, 50, 60), -100, dtype=np.int16)
    lbl = np.zeros(img.shape, dtype=np.uint8)
    lbl[10:20, 10:30, 20:40] = 6
    np.save(tmp_path / "amos_0001.img.npy", img)
    np.save(tmp_path / "amos_0001.lbl.npy", lbl)
    ds = AMOSDataset(["amos_0001"], tmp_path, patch=(32, 64, 48), samples_per_case=3,
                     train=True, strong_aug=True)
    item = ds[0]
    assert item["image"].shape == (1, 32, 64, 48) and item["label"].shape == (32, 64, 48)
    assert item["present"][0] and item["present"].dtype == torch.bool
    val = AMOSDataset(["amos_0001"], tmp_path, patch=(32, 32, 32), samples_per_case=2, train=False)
    assert torch.equal(val[1]["image"], val[1]["image"]) and val[1]["label"].max() == 6

    ids = [f"amos_{i:04d}" for i in range(23)]
    folds = make_folds(ids, 5, seed=42)
    vals = [v for _, v in folds]
    assert sorted(sum(vals, [])) == ids and all(4 <= len(v) <= 5 for v in vals)
    assert folds == make_folds(list(reversed(ids)), 5, seed=42)


def test_largest_only_drops_far_fragment_for_listed_organs():
    from src.infer.postprocess import clean_label

    label = np.zeros((30, 30, 30), dtype=np.uint8)
    label[2:12, 2:12, 2:12] = 2          # 主体 1000 体素
    label[20:25, 20:25, 20:25] = 2       # 远处 125 体素（> 2%，相对阈值会保留）
    kept, _ = clean_label(label)
    assert (kept == 2).sum() == 1125
    solid, _ = clean_label(label, largest_only={2})
    assert (solid == 2).sum() == 1000
