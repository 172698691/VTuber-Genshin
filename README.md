# INF573 - Image Analysis and Computer Vision

This is the final project of course **INF573 - Image Analysis and Computer Vision**. We implement a Genshin VTuber base on [VTuber-MomoseHiyori](https://github.com/KennardWang/VTuber-MomoseHiyori) by KennardWang.

## Notes

This program is implemented base on Windows 10 x64 System. Using Python 3.8 and Unity 2019.4.1f1. The Unity Engine use port number `14514`.

## Requirements

Make sure you have the required dependencies installed by running (Recommended to use a conda virtual environment):
```bash
pip install -r requirements.txt
```

## Command-line Arguments

- `--cam`: Specify the index of camera. (Default: 0)
- `--debug`: Whether show image and marks or not. (Default: False)
- `--connect`: Whether connect to unity character or not. (Default: False)
- `--video`: Specify the path to the video file (if not using camera). (Default: None)

## Run

**Step 1 : Test camera**

Run the following code, you should see your face with marks.
```bash
python main.py --debug
```
Press `q` to exit the program.

<p align = "center">
    <img src = "img_source\face.gif" width = "240px" height = "240px"/>
</p>

**Step 2 : Connect Unity**

First, run the Unity executable file `Genshin-vtuber.exe`.

Then run with camera
```bash
python main.py --connect
```

or

run with video (replace `/path_of_video` with your actual paths)
```bash
python main.py --connect --video /path_of_video
```

Press `q` to exit the program.

<p align = "center">
    <img src = "img_source\result.gif" width = "630px" height = "270px"/>
</p>
