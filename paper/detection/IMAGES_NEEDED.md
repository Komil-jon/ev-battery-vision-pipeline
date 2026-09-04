# Images needed for the qualitative figures

**Status: the divergent-convention half is DONE.** 143 MTech frames with matching
labels were found at `~/EV-battery-detection/data/detector`, and
`fig_convention_examples.pdf` is rendered from them.

## Still needed: consensus-convention frames

The figure is a side-by-side, and the consensus side is missing. Any *one* of these
four is enough; two or three is better. Filenames must be unchanged so they match the
label files already in the repo.

```
train/roboflow_ev-battery-component-detection-gqljq_Skywell-BE11-2021-_raw_frame_0532_jpg.rf.4abd3e8c13f9bd57aa8d066eb0f24281.<jpg|png>
    1 module, 4 busbars, median module scale 0.948
train/roboflow_ev-battery-components-edfw3_image57_jpg.rf.da31f580f9598f61f3bf5957b8c149a2.<jpg|png>
    5 modules, 1 busbar, median module scale 0.153
train/bmw_i3_50_png.rf.98958f84b2f9d6ec6a9c4e8080db1927.<jpg|png>
    2 modules, median module scale 0.423
train/ue_rav4_module_1726785169_png.rf.4bf7ce5d1c19d9eab66ca767fb203131.<jpg|png>
    1 module, median module scale 0.552
```

Any other frame from `gqljq`, `edfw3`, `bmw_i3`, `ue_rav4`, `battery_comp`,
`final_mobilenet` or `ybmvt` works too, as long as a label file of the same stem
exists under `data/labels_release/detector/`.

## Where to get them

Each source's Roboflow Universe project, with its URL and licence, is listed in
`docs/DATASETS.md`. Or re-download in bulk:

```bash
python scripts/data_prep/download_external_datasets.py
```

## Rebuilding the figure once they are present

```bash
python paper/detection/render_convention_figure.py \
    --image-dirs ~/EV-battery-detection/data/detector ~/path/to/new/images
```
