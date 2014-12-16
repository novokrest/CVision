__author__ = 'k.novokreshchenov'

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from Tkinter import *
import io_settings as io


def generate_labels(parent, keys):
    labels = {}
    for k in keys:
        labels[k] = Label(parent, text=k,
                          fg='white', font='arial 14')
    return labels


def generate_scales(parent, key2range, command=None):
    scales = {}
    for key, range_ in key2range.items():
        (start, end, step) = range_
        scales[key] = Scale(parent, orient=HORIZONTAL, length=400, label=key,
                            from_=start, to=end, resolution=1,
                            command=command)
    return scales


def init_frame_with_settings(root, name, settings, scale_command=None):
    frame = Frame(root, height=200, width=200, bd=5)
    label = Label(frame, text=name)
    keys = settings.keys()
    settings_scales = generate_scales(frame, settings, scale_command)
    label.grid(row=1)
    i = 2
    for key in keys:
        settings_scales[key].grid(row=i)
        i += 1
    return frame, settings_scales

def scales_values(scales):
    values = {}
    for key, scale in scales.items():
        values[key] = scale.get()
    return values

def convert_to_odd(n):
    if n % 2 == 0:
        return n + 1
    return n

class App:
    def __init__(self, img):
        self.src_img = img; self.g_img = img; self.l_img = img; self.result = img
        self.root = Tk(); self.root.wm_title('CompVision_HW_1')

        self.gaussian_frame, self.gaussian_scales = init_frame_with_settings(self.root, 'Gaussian',
                                                                             gaussian_settings,
                                                                             self.update_gaussian)
        self.laplacian_frame, self.laplacian_scales = init_frame_with_settings(self.root, 'Laplacian',
                                                                               laplacian_settings,
                                                                               self.update_laplacian)
        self.dilate_frame, self.dilate_scales = init_frame_with_settings(self.root, 'Dilate',
                                                                         dilate_settings)
        self.erode_frame, self.erode_scales = init_frame_with_settings(self.root, 'Erode',
                                                                       erode_settings)
        self.dilate_button = Button(self.root, text="DILATE", command=self.dilate)
        self.erode_button = Button(self.root, text="ERODE", command=self.erode)
        self.contours_button = Button(self.root, text="CONTOURS", command=self.draw_contours)
        self.save_button = Button(self.root, text="SAVE", command=self.save_settings)
        self.show_button = Button(self.root, text="SHOW RESULT", command=self.show_result)
        self.refresh_button = Button(self.root, text="REFRESH", command=self.refresh)

        self.gaussian_frame.grid(row=2, column=1)
        self.laplacian_frame.grid(row=2, column=2)
        self.dilate_frame.grid(row=4, column=1)
        self.erode_frame.grid(row=4, column=2)
        self.dilate_button.grid(row=5, column = 1)
        self.erode_button.grid(row=5, column=2)
        self.contours_button.grid(row=6, column=1, columnspan=2)
        self.save_button.grid(row=7, column=1, columnspan=2)
        self.refresh_button.grid(row=8, column=1, columnspan=2)

    def load_settings(self):
        self.settings = io.read_settings(io.SETTINGS_PATH)
        self.gaussian_scales['ksizeX'].set(self.settings['gaussian_ksizeX'])
        self.gaussian_scales['ksizeY'].set(self.settings['gaussian_ksizeY'])
        self.gaussian_scales['sigmaX'].set(self.settings['gaussian_sigmaX'])
        self.gaussian_scales['sigmaY'].set(self.settings['gaussian_sigmaY'])
        self.laplacian_scales['ksize'].set(self.settings['laplacian_ksize'])
        self.laplacian_scales['scale'].set(self.settings['laplacian_scale'])
        self.dilate_scales['kernelX'].set(self.settings['dilate_kernelX'])
        self.dilate_scales['kernelY'].set(self.settings['dilate_kernelY'])
        self.erode_scales['kernelX'].set(self.settings['erode_kernelX'])
        self.erode_scales['kernelY'].set(self.settings['erode_kernelY'])

    def save_settings(self):
        self.settings['gaussian_ksizeX'] = self.gaussian_scales['ksizeX'].get()
        self.settings['gaussian_ksizeY'] = self.gaussian_scales['ksizeY'].get()
        self.settings['gaussian_sigmaX'] = self.gaussian_scales['sigmaX'].get()
        self.settings['gaussian_sigmaY'] = self.gaussian_scales['sigmaY'].get()
        self.settings['laplacian_ksize'] = self.laplacian_scales['ksize'].get()
        self.settings['laplacian_scale'] = self.laplacian_scales['scale'].get()
        self.settings['dilate_kernelX'] = self.dilate_scales['kernelX'].get()
        self.settings['dilate_kernelY'] = self.dilate_scales['kernelY'].get()
        self.settings['erode_kernelX'] = self.erode_scales['kernelX'].get()
        self.settings['erode_kernelY'] = self.erode_scales['kernelY'].get()
        io.write_settings(io.SETTINGS_PATH, self.settings)

    def start(self):
        self.load_settings()
        self.root.mainloop()

    def update_gaussian(self, event):
        values = scales_values(self.gaussian_scales)
        values['ksizeX'] = convert_to_odd(values['ksizeX'])
        values['ksizeY'] = convert_to_odd(values['ksizeY'])
        try:
            self.g_img = cv2.GaussianBlur(src=self.src_img,
                                          ksize=(values['ksizeX'], values['ksizeY']),
                                          sigmaX=values['sigmaX'])
        except Exception as e:
            print(e.message)
        self.update_laplacian(event)

    def update_laplacian(self, event):
        values = scales_values(self.laplacian_scales)
        values['ksize'] = convert_to_odd(values['ksize'])
        try:
            self.l_img = cv2.Laplacian(src=self.g_img,
                                       ddepth=cv2.CV_32F,
                                       ksize=values['ksize'],
                                       scale=values['scale'])
        except Exception as e:
            print(e.message)
        ret, self.result = cv2.threshold(self.l_img, 0, 255, cv2.THRESH_BINARY)
        self.save_result()

    def dilate(self):
        x = self.dilate_scales['kernelX'].get()
        y = self.dilate_scales['kernelY'].get()
        kernel = np.ones((x, y), np.uint8)
        self.result = cv2.dilate(self.result, kernel, iterations=1)
        self.save_result()

    def erode(self):
        x = self.erode_scales['kernelX'].get()
        y = self.erode_scales['kernelY'].get()
        kernel = np.ones((x, y), np.uint8)
        self.result = cv2.erode(self.result, kernel, iterations=1)
        self.save_result()

    def draw_contours(self):
        ret, self.result = cv2.threshold(self.result, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(self.result), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(self.result, (x, y), (x+w, y+h), (255, 0, 0), -1)
        self.save_result()

    def draw_curve_contours(self):
        ret, self.result = cv2.threshold(self.result, 0, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(cv2.convertScaleAbs(self.result), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.result, contours, -1, (255,0,0), 3)

    def draw_image(self, img, column):
        figure = Figure((4,4))
        plot = figure.add_subplot(1, 1, 1)
        plot.axes.get_xaxis().set_visible(False), plot.axes.get_yaxis().set_visible(False)
        plot.imshow(img)

        canvas = FigureCanvasTkAgg(figure, master=self.root)
        canvas.show()
        canvas.get_tk_widget().grid(row=1, column=column)

    def draw_src(self):
        self.draw_image(self.src_img, column=1)

    def draw_result(self):
        self.draw_image(self.result, column=2)

    def save_result(self):
        cv2.imwrite('result.bmp', self.result)

    def show_result(self):
        cv2.imshow("result", self.result)

    def refresh(self):
        self.result = self.l_img
        self.save_result()


IMG_PATH = 'text.bmp'
gaussian_settings = {'ksizeX': (1, 51, 1),
                     'ksizeY': (1, 51, 1),
                     'sigmaX': (0, 100, 1),
                     'sigmaY': (0, 100, 1)}
laplacian_settings = {'ksize': (1, 31, 1),
                      'scale': (0, 100, 1)}
dilate_settings = {'kernelX': (0, 10, 1),
                   'kernelY': (0, 10, 1)}
erode_settings = {'kernelX': (0, 10, 1),
                  'kernelY': (0, 10, 1)}

if __name__ == '__main__':
    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
    app = App(img)
    app.start()