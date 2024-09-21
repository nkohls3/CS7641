import os

from dataloader import Image


def walk_through_images(dir_path):
    print("Press q/Q to quit.")
    print("Press any other button to walk through images.")

    for _, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".jpg"):
                path = dir_path + file
                img = Image(path)
                ui = img.show_waitkey()

                # break if q/Q entered: https://www.asciitable.com/
                if ui == 81 or ui == 113:
                    break
