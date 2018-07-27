import tkinter as tk
from PIL import ImageTk, Image
import os
import sys

class Application(tk.Frame):
    def __init__(self, log_file, im_root_path, master = None):
        tk.Frame.__init__(self, master)

        self.img_copy = None
        self.log_file = log_file

        self.seen = {}
        try:
            with open(self.log_file, 'r+') as f:
                text = f.read()
                lines = text.split('\n')
                files = [line.split(',')[0] for line in lines]
                self.seen = {img:True for img in files}
        except:
            print("log file does not exist, will be created")

        self.photos = self.find_photos(im_root_path)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.panel = tk.Label(self)
        self.panel.pack(side = 'top', fill = 'both', expand = 'yes')
        self.change_image()
        self.bind('<Configure>', self.resize_image)
        self.bind_all('<Key>', self.key)
        self.bind_all('<Escape>', self.close)

    def resize_image(self, event):
        self.image = self.img_copy.resize((event.width, event.height))

        self.img = ImageTk.PhotoImage(self.image)
        self.panel.configure(image = self.img)

    def find_photos(self, base_path):
        for path, dirs, files in os.walk(base_path):
            dirs.sort(reverse=True)
            files.sort(reverse=True)
            for fi in files:
                if fi.endswith('.jpg') and not self.seen.get(os.path.join(path, fi), False):
                    yield os.path.join(path, fi)

    def change_image(self):
        self.cur_img = next(self.photos)
        print(self.cur_img)
        if self.cur_img is None:
            self.master.destroy()

        self.image = Image.open(self.cur_img)
        self.img_copy = self.image.copy()

        self.img = ImageTk.PhotoImage(self.image)
        self.panel.configure(image = self.img)

    def ignore(self, event):
        return "break"

    def close(self, event):
        sys.exit()

    def key(self, event):
        self.master.bind_all('<Key>', self.ignore)

        input = event.char
        with open(self.log_file, 'a+') as log:
            log.write(self.cur_img + ',' + input + '\n')
        self.change_image()
        print(event.char)
        self.master.bind_all('<Key>', self.key)



if __name__ == "__main__":
    root = tk.Tk()
    root.title('Photo Labeler')
    root.geometry('500x600')
    root.configure(background='black')

    log_file = sys.argv[1]
    photo_dir = sys.argv[2]
    app = Application(log_file , photo_dir, master = root)
    app.pack(fill='both', expand='yes')
    app.mainloop()
