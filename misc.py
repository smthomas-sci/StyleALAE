"""
A bunch of random functions for visualisations

Author: Simon Thomas
Date: 03-08-2020

"""
import os
import shutil


def prepare_progress_for_gif(sizes, image_dir, output_dir):
    """
    Moves images from 4x4 -> 128x128 etc to output_dir, ordered so as
    to be converted to gif or mp4.
    :param sizes: list of sizes  e.g. [4, 8, 16, 32]
    :param image_dir: the parent directory of outputs e.g. parent/{size}x{size}/
    :param output_dir: the output directory
    :return: None
    """
    count = 0
    print("moving progress images ...", end=" ")
    for size in sizes:

        # Get files
        size_dir = os.path.join(image_dir, f"./{size}x{size}/")
        files = os.listdir(size_dir)

        # Sort files
        merged = []
        straight = []
        for file in files:
            if "merge" in file:
                merged.append(file)
            else:
                straight.append(file)
        merged.sort()
        straight.sort()

        # Move files
        for file in merged + straight:
            src = os.path.join(size_dir, file)
            dst = os.path.join(output_dir, f"{count:04}.jpg")
            shutil.copy(src, dst)
            count += 1
    print("done.")


