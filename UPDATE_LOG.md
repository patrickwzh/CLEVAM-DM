## 2025.05.12

by wzh

-   Update meeting note
-   Reconstruct repo
-   Push tips:
    -   Avoid using `git add .`, only pick specific files to stage.
    -   Stage config files only if necessary (we don't want our local `video_path` to be overwritten by yours).
    -   Ensure code and directory structure is maintained, write `utils.py` if necessary.
    -   Write docstrings for interface-level functions and classes.

## 2025.04.27

by lhz

-   Pipeline init
-   directory Structure

```
--- {INPUT_DIR}
    --- input_vedio.mp4
--- {OUTPUT_DIR}
    --- pictures
        --- keyframes
            --- keyframe_0001.png
            --- keyframe_0002.png
            ...
        --- gapframes
            --- gapframe_0001
                --- gapframe_0001_0001.png
                --- gapframe_0001_0002.png
                ...
            --- gapframe_0002
                --- gapframe_0002_0001.png
                --- gapframe_0002_0002.png
    --- output_video.mp4
```

## 2025.04.27

by wzh

-   Inital commit
