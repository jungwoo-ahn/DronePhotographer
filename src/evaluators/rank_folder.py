"""Rank + visualize images in a plain folder.

This script is for quick inspection when you have an image folder like
`assets/movie/` (or any folder of PNG/JPG frames) and you want:

1) Run GroundingDINO detection (optional) for a text prompt.
2) Run one or more evaluators from `src/evaluators/evaluator.py`.
3) Save an `annotations.json` (similar shape to render runs).
4) Save one or more visualization grids with score overlays.

Unlike `tools/visualize_rankings.py`, this operates directly on a folder of
images (no pre-existing `annotations.json` required).

Example:

```bash
python -m src.evaluators.rank_folder \
  --image_dir assets/movie \
  --prompt "a man" \
  --metrics composition qalign \
  --out_dir outputs/movie_rank \
  --limit 30
```
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
	import matplotlib

	matplotlib.use("Agg")  # headless friendly
	import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
	raise SystemExit(
		"matplotlib is required. Install it with `pip install matplotlib`."
	) from exc


from src.evaluators.evaluator import (
	BreathingSpaceEvaluator,
	BrightnessEvaluator,
	ColorContrastEvaluator,
	CompositeEvaluator,
	EvaluationResult,
	LaplacianEvaluator,
	QAlignEvaluator,
	RuleOfThirdsEvaluator,
	StddevEvaluator,
	SubjectSizeEvaluator,
	default_composition_evaluator,
)


DEFAULT_PROMPT = "a man"
DEFAULT_COLS = 5
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


_DETECTION_BASED_METRICS = {
	"composition",
	"color_contrast",
	"subject_size",
	"rule_of_thirds",
	"breathing_space",
}


def _is_detection_based_metric(metric: str) -> bool:
	return str(metric).strip().lower() in _DETECTION_BASED_METRICS


def _draw_detections_bgr(image_bgr, detections: Sequence[Dict[str, Any]]):
	"""Draw bbox + label/score overlays onto a copy of BGR image."""
	try:
		import cv2
	except Exception:
		return image_bgr

	if image_bgr is None or not hasattr(image_bgr, "shape"):
		return image_bgr

	out = image_bgr.copy()
	h, w = out.shape[:2]

	def _clip(v, lo, hi):
		return max(lo, min(hi, v))

	if not detections:
		# Add an explicit "no detections" overlay.
		cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), thickness=-1)
		cv2.putText(
			out,
			"NO DETECTIONS",
			(8, 24),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(0, 0, 255),
			2,
			lineType=cv2.LINE_AA,
		)
		return out

	for d in detections:
		if not isinstance(d, dict):
			continue
		box = d.get("bbox_xyxy") or d.get("box_xyxy") or d.get("bbox")
		if not box or len(box) != 4:
			continue
		try:
			x1, y1, x2, y2 = [int(round(float(v))) for v in box]
		except Exception:
			continue
		x1 = _clip(x1, 0, w - 1)
		x2 = _clip(x2, 0, w - 1)
		y1 = _clip(y1, 0, h - 1)
		y2 = _clip(y2, 0, h - 1)
		if x2 <= x1 or y2 <= y1:
			continue

		label = d.get("label")
		score = d.get("score")
		text = ""
		if label is not None:
			text += str(label)
		if score is not None:
			try:
				text += f" {float(score):.2f}" if text else f"{float(score):.2f}"
			except Exception:
				pass

		color = (0, 255, 0)
		cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness=2)
		if text:
			# Solid bg for readability.
			(tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
			ty1 = max(0, y1 - th - baseline - 4)
			cv2.rectangle(out, (x1, ty1), (x1 + tw + 6, ty1 + th + baseline + 4), (0, 0, 0), -1)
			cv2.putText(
				out,
				text,
				(x1 + 3, ty1 + th + 2),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5,
				(255, 255, 255),
				1,
				lineType=cv2.LINE_AA,
			)

	return out


def _draw_rule_of_thirds_lines_bgr(image_bgr):
	"""Draw rule-of-thirds grid lines on a copy of BGR image."""
	try:
		import cv2
	except Exception:
		return image_bgr

	if image_bgr is None or not hasattr(image_bgr, "shape"):
		return image_bgr

	out = image_bgr.copy()
	h, w = out.shape[:2]

	# 2 vertical lines, 2 horizontal lines
	xs = [int(round(w / 3.0)), int(round(2.0 * w / 3.0))]
	ys = [int(round(h / 3.0)), int(round(2.0 * h / 3.0))]

	color = (0, 215, 255)  # orange-ish (BGR)
	thickness = 2
	alpha = 0.35
	
	overlay = out.copy()
	for x in xs:
		cv2.line(overlay, (x, 0), (x, h - 1), color, thickness=thickness, lineType=cv2.LINE_AA)
	for y in ys:
		cv2.line(overlay, (0, y), (w - 1, y), color, thickness=thickness, lineType=cv2.LINE_AA)

	out = cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)
	return out


def _bgr_to_float_rgb(image_bgr):
	"""Convert BGR uint8 image to RGB float [0,1] for matplotlib."""
	try:
		import cv2
		rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)
	except Exception:
		# best-effort fallback (may look wrong if BGR)
		arr = image_bgr
		if arr is None:
			return np.full((64, 64, 3), 0.8, dtype=float)
		arr = arr.astype(np.float32)
		if arr.max() > 1.0:
			arr = arr / 255.0
		return arr.clip(0.0, 1.0)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run detection + evaluator metrics on an image folder and visualize rankings."
	)
	parser.add_argument(
		"--image_dir",
		type=Path,
		required=True,
		help="Folder containing images (png/jpg/...).",
	)
	parser.add_argument(
		"--out_dir",
		type=Path,
		default=None,
		help="Output directory. Default: <image_dir>/rank_out",
	)
	parser.add_argument(
		"--prompt",
		type=str,
		default=DEFAULT_PROMPT,
		help=f"Detection prompt (default: {DEFAULT_PROMPT!r}).",
	)
	parser.add_argument(
		"--metrics",
		nargs="+",
		default=["composition"],
		help=(
			"Which metrics to compute. Supported: composition, color_contrast, subject_size, "
			"rule_of_thirds, breathing_space, brightness, laplacian, stddev, qalign"
		),
	)
	parser.add_argument(
		"--no_detect",
		action="store_true",
		help="Skip detection entirely (composition-like metrics will score as 0).",
	)
	parser.add_argument(
		"--detector_config",
		type=Path,
		default=_REPO_ROOT / "repos/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
		help="GroundingDINO config path.",
	)
	parser.add_argument(
		"--detector_weights",
		type=Path,
		default=_REPO_ROOT / "repos/GroundingDINO/weights/groundingdino_swint_ogc.pth",
		help="GroundingDINO weights path.",
	)
	parser.add_argument("--device", type=str, default=None, help="Detector device: cuda/cpu")
	parser.add_argument("--box_threshold", type=float, default=0.35)
	parser.add_argument("--text_threshold", type=float, default=0.25)
	parser.add_argument(
		"--skip_first",
		type=int,
		default=0,
		help="Skip the first N images after sorting (cut from front).",
	)
	parser.add_argument(
		"--max_images",
		type=int,
		default=None,
		help="Maximum number of images to process (after --skip_first).",
	)
	parser.add_argument(
		"--cols",
		type=int,
		default=DEFAULT_COLS,
		help=f"Number of columns in output grids (default: {DEFAULT_COLS}).",
	)
	parser.add_argument(
		"--limit",
		type=int,
		default=None,
		help="Limit number of images shown in ranking grids (after sorting).",
	)
	parser.add_argument(
		"--ascending",
		action="store_true",
		help="Sort ascending (default: descending).",
	)
	parser.add_argument(
		"--write_all_grids",
		action="store_true",
		help="Write per-metric grids (default: on). Use --combined_only to disable.",
	)
	parser.add_argument(
		"--combined_only",
		action="store_true",
		help="Only write the combined grid (disables per-metric grids).",
	)
	parser.add_argument(
		"--no_viz",
		action="store_true",
		help="Skip writing grid images (still writes annotations.json).",
	)
	return parser.parse_args(list(argv) if argv is not None else None)


def _flatten_scores_into_entry(entry: Dict[str, Any], evaluation: Dict[str, Any]) -> None:
	"""Populate entry with flat score_* fields for all known metrics."""
	name = str(evaluation.get("name", "evaluation"))
	score = _safe_float(evaluation.get("score"))
	if score is not None:
		entry[f"score_{name}"] = float(score)

	details = evaluation.get("details") or {}
	components = details.get("components")
	if isinstance(components, list):
		for comp in components:
			if not isinstance(comp, dict):
				continue
			cname = comp.get("name")
			cscore = _safe_float(comp.get("score"))
			if cname is None or cscore is None:
				continue
			entry[f"score_{str(cname)}"] = float(cscore)


def _list_images(image_dir: Path) -> List[Path]:
	if not image_dir.exists() or not image_dir.is_dir():
		raise SystemExit(f"image_dir not found or not a directory: {image_dir}")
	images = [p for p in sorted(image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
	if not images:
		raise SystemExit(f"No images found in {image_dir} (supported: {sorted(SUPPORTED_EXTS)})")
	return images


def _build_evaluator(metric_names: Sequence[str]) -> CompositeEvaluator | Any:
	# Kept for backwards compatibility if someone imports it, but rank_folder
	# now evaluates metrics one-by-one (see _build_evaluators_map).
	return default_composition_evaluator()


def _normalize_metric_list(metrics: Sequence[str]) -> List[str]:
	out: List[str] = []
	for m in metrics:
		key = str(m).strip().lower()
		if key == "rule_of_thrids":
			key = "rule_of_thirds"
		out.append(key)
	return out


def _build_evaluators_map(metric_names: Sequence[str]) -> Dict[str, Any]:
	metric_builders = {
		"color_contrast": ColorContrastEvaluator,
		"subject_size": SubjectSizeEvaluator,
		"rule_of_thirds": RuleOfThirdsEvaluator,
		"breathing_space": BreathingSpaceEvaluator,
		"brightness": BrightnessEvaluator,
		"laplacian": LaplacianEvaluator,
		"stddev": StddevEvaluator,
		"qalign": QAlignEvaluator,
		"composition": default_composition_evaluator,
	}

	metrics = _normalize_metric_list(metric_names)
	unknown = [m for m in metrics if m not in metric_builders]
	if unknown:
		raise SystemExit(
			f"Unknown metric(s): {unknown}. Supported: {', '.join(sorted(metric_builders.keys()))}"
		)

	out: Dict[str, Any] = {}
	for m in metrics:
		builder = metric_builders[m]
		out[m] = builder() if callable(builder) else builder
	return out


def _load_bgr(path: Path):
	import cv2

	img = cv2.imread(str(path))
	return img


def _to_jsonable_detection(d: Any) -> Dict[str, Any]:
	if isinstance(d, dict):
		return dict(d)
	if hasattr(d, "as_dict"):
		return d.as_dict()
	box = getattr(d, "box_xyxy", None) or getattr(d, "bbox_xyxy", None)
	score = getattr(d, "score", None)
	label = getattr(d, "label", None)
	payload: Dict[str, Any] = {}
	if label is not None:
		payload["label"] = str(label)
	if score is not None:
		payload["score"] = float(score)
	if box is not None:
		x1, y1, x2, y2 = [float(v) for v in box]
		payload["bbox_xyxy"] = [x1, y1, x2, y2]
	return payload


def _eval_to_dict(res: EvaluationResult) -> Dict[str, Any]:
	try:
		return asdict(res)
	except Exception:
		return {"name": res.name, "score": float(res.score), "details": dict(res.details)}


def _safe_float(value: Any) -> Optional[float]:
	try:
		num = float(value)
	except (TypeError, ValueError):
		return None
	if math.isnan(num) or math.isinf(num):
		return None
	return num


def _plot_ranked_grid(
	items: List[Dict[str, Any]],
	title: str,
	out_path: Path,
	cols: int,
	ascending: bool,
) -> Path:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	cols = int(cols) if cols and cols > 0 else DEFAULT_COLS
	total = len(items)
	rows = math.ceil(total / cols) if total > 0 else 1

	fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
	axes = np.atleast_2d(axes).flatten()
	for ax in axes:
		ax.set_axis_off()

	for idx, entry in enumerate(items):
		ax = axes[idx]
		# Prefer pre-rendered RGB image (e.g., detection-annotated) if provided.
		img = entry.get("viz_rgb")
		if img is None:
			path: Path = Path(entry["path"])
			try:
				img = plt.imread(path)
			except Exception:
				img = np.full((64, 64, 3), 0.8, dtype=float)
		ax.imshow(img)

		score_parts = []
		for k, v in entry.get("scores", {}).items():
			if v is None:
				continue
			score_parts.append(f"{k}={v:.3f}")
		score_text = " | ".join(score_parts) if score_parts else "(no scores)"

		# Avoid set_title() (it triggers tick/layout calculations and can be slow)
		ax.text(
			0.02,
			0.98,
			f"#{idx+1}",
			fontsize=8,
			color="white",
			ha="left",
			va="top",
			transform=ax.transAxes,
			bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
		)
		ax.text(
			0.02,
			0.90,
			score_text,
			fontsize=7,
			color="white",
			ha="left",
			va="top",
			transform=ax.transAxes,
			bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
		)
		ax.text(
			0.02,
			0.08,
			Path(entry.get("path")).name,
			fontsize=7,
			color="white",
			ha="left",
			va="bottom",
			transform=ax.transAxes,
			bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
		)

	fig.suptitle(
		f"{title} (sorted {'ascending' if ascending else 'descending'}) — n={total}",
		fontsize=12,
	)
	fig.tight_layout(rect=[0, 0, 1, 0.96])
	fig.savefig(out_path, dpi=220)
	plt.close(fig)
	return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
	args = _parse_args(argv)
	image_dir: Path = args.image_dir
	out_dir: Path = args.out_dir or (image_dir / "rank_out")
	out_dir.mkdir(parents=True, exist_ok=True)
	viz_dir = out_dir / "viz"
	viz_dir.mkdir(parents=True, exist_ok=True)

	images = _list_images(image_dir)
	skip_first = int(args.skip_first or 0)
	if skip_first < 0:
		skip_first = 0
	if skip_first:
		images = images[skip_first:]
	max_images = args.max_images
	if max_images is not None:
		max_images = int(max_images)
		if max_images > 0:
			images = images[:max_images]
		elif max_images <= 0:
			images = []
	if not images:
		raise SystemExit("No images to process after applying --skip_first/--max_images.")
	evaluators = _build_evaluators_map(args.metrics)
	metric_order = list(evaluators.keys())

	detector = None
	if not args.no_detect:
		try:
			from src.detectors.detector import GroundingDINODetector  # lazy import
		except Exception as exc:
			raise SystemExit(
				"Failed to import GroundingDINO detector. "
				"Either install GroundingDINO (editable install in repos/GroundingDINO) "
				"or run with --no_detect.\n"
				f"Original error: {exc}"
			) from exc

		detector_config = Path(args.detector_config)
		detector_weights = Path(args.detector_weights)
		if not detector_config.exists():
			raise SystemExit(f"Detector config not found: {detector_config}")
		if not detector_weights.exists():
			raise SystemExit(f"Detector weights not found: {detector_weights}")
		detector = GroundingDINODetector(
			model_config_path=str(detector_config),
			model_checkpoint_path=str(detector_weights),
			device=args.device,
		)

	annotations: List[Dict[str, Any]] = []
	for path in images:
		image_bgr = _load_bgr(path)
		detections: List[Any] = []
		if detector is not None:
			try:
				detections = detector.detect_file(
					image_path=path,
					caption=args.prompt,
					box_threshold=float(args.box_threshold),
					text_threshold=float(args.text_threshold),
				)
			except Exception as exc:
				detections = []
				print(f"[warn] detect failed for {path}: {exc}")

		entry: Dict[str, Any] = {
			"image": str(path.relative_to(image_dir)),
			"image_abs": str(path),
			"prompt": str(args.prompt),
			"detections": [_to_jsonable_detection(d) for d in detections],
			"evaluations": {},
		}

		# Evaluate each metric separately with the correct input type.
		# - composition / bbox-based heuristics need an image array with .shape
		# - qalign can take BGR array (it internally converts to PIL)
		for metric_name, evaluator in evaluators.items():
			try:
				res = evaluator.evaluate(image_bgr, detections)
			except Exception as exc:
				res = EvaluationResult(
					name=str(getattr(evaluator, "name", metric_name)),
					score=0.0,
					details={"reason": "eval_failed", "error": str(exc)},
				)
			res_dict = _eval_to_dict(res)
			entry["evaluations"][metric_name] = res_dict
			_flatten_scores_into_entry(entry, res_dict)

		# Choose a primary score for sorting/ranking.
		primary = None
		for pref in ("qalign", "composition"):
			v = _safe_float(entry.get(f"score_{pref}"))
			if v is not None:
				primary = v
				break
		if primary is None and metric_order:
			primary = _safe_float(entry.get(f"score_{metric_order[0]}"))
		entry["score_primary"] = float(primary) if primary is not None else 0.0
		annotations.append(entry)

	annotations_path = out_dir / "annotations.json"
	with annotations_path.open("w", encoding="utf-8") as f:
		json.dump(annotations, f, indent=2)

	# Build ranking items
	items: List[Dict[str, Any]] = []
	for a in annotations:
		dets_for_viz = a.get("detections") or []
		use_det_viz = any(_is_detection_based_metric(m) for m in metric_order)
		viz_rgb = None
		if use_det_viz:
			img_bgr = _load_bgr(Path(a["image_abs"]))
			if img_bgr is not None:
				viz_bgr = _draw_detections_bgr(img_bgr, dets_for_viz)
				# If we're computing rule_of_thirds, add thirds grid overlay in combined view.
				if any(str(m).strip().lower() == "rule_of_thirds" for m in metric_order):
					viz_bgr = _draw_rule_of_thirds_lines_bgr(viz_bgr)
				viz_rgb = _bgr_to_float_rgb(viz_bgr)

		score_map: Dict[str, Optional[float]] = {}
		# Display a compact set of scores in the combined grid.
		for m in metric_order:
			score_map[m] = _safe_float(a.get(f"score_{m}"))
		# If composition was computed, also show its components.
		comp_eval = (a.get("evaluations") or {}).get("composition")
		if isinstance(comp_eval, dict):
			components = (comp_eval.get("details") or {}).get("components")
			if isinstance(components, list):
				for c in components:
					if not isinstance(c, dict):
						continue
					nm = c.get("name")
					sc = _safe_float(c.get("score"))
					if nm is not None:
						score_map[str(nm)] = sc

		items.append(
			{
				"path": Path(a["image_abs"]),
				"rel": a.get("image"),
				"primary": _safe_float(a.get("score_primary")),
				"scores": score_map,
				"viz_rgb": viz_rgb,
			}
		)

	items.sort(key=lambda x: (x["primary"] is None, x["primary"]), reverse=not args.ascending)
	if args.limit is not None and args.limit > 0:
		items = items[: int(args.limit)]

	combined_path = viz_dir / "rankings_combined.png"
	if not args.no_viz:
		_plot_ranked_grid(
			items,
			title=f"{Path(image_dir).name} | prompt={args.prompt} | metrics={','.join(args.metrics)}",
			out_path=combined_path,
			cols=int(args.cols),
			ascending=bool(args.ascending),
		)

	write_all = (bool(args.write_all_grids) or (not bool(args.combined_only))) and (not bool(args.no_viz))
	if write_all:
		# Write one grid per score key we can find.
		score_keys = sorted({k for a in annotations for k in a.keys() if str(k).startswith("score_")})
		for key in score_keys:
			metric = key.replace("score_", "")
			metric_items = []
			for a in annotations:
				viz_rgb = None
				if _is_detection_based_metric(metric):
					img_bgr = _load_bgr(Path(a["image_abs"]))
					if img_bgr is not None:
						viz_bgr = _draw_detections_bgr(img_bgr, a.get("detections") or [])
						if str(metric).strip().lower() == "rule_of_thirds":
							viz_bgr = _draw_rule_of_thirds_lines_bgr(viz_bgr)
						viz_rgb = _bgr_to_float_rgb(viz_bgr)
				metric_items.append(
					{
						"path": Path(a["image_abs"]),
						"rel": a.get("image"),
						"primary": _safe_float(a.get(key)),
						"scores": {metric: _safe_float(a.get(key))},
						"viz_rgb": viz_rgb,
					}
				)
			metric_items.sort(
				key=lambda x: (x["primary"] is None, x["primary"]), reverse=not args.ascending
			)
			if args.limit is not None and args.limit > 0:
				metric_items = metric_items[: int(args.limit)]
			_plot_ranked_grid(
				metric_items,
				title=f"{Path(image_dir).name} | {metric}",
				out_path=viz_dir / f"rankings_{metric}.png",
				cols=int(args.cols),
				ascending=bool(args.ascending),
			)

	print(f"Wrote: {annotations_path}")
	if not args.no_viz:
		print(f"Wrote: {combined_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())


'''
CUDA_VISIBLE_DEVICES=7 python -m src.evaluators.rank_folder \
  --image_dir assets/movie \
  --prompt "face" \
  --metrics composition color_contrast subject_size rule_of_thirds breathing_space brightness laplacian stddev qalign \
  --skip_first 100 \
  --max_images 150 \
  --out_dir outputs/movie_rank_face \
  --device cuda 
  
#   --limit 30
'''
