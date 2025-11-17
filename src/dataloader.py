import cv2
import numpy as np
import copy as c
import functools
# from dataloader import Trial


class Trial:
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    UNDO_QUEUE_LENGTH = 3

    def __init__(self, video_path:str):
        self.frames = self.load_mov_as_array(video_path)
        self.__original = self.frames
        self.shape = self[0].shape
        self.__undo_list = []
        

    def load_mov_as_array(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
        frames = []
        while True:
            ret, frame = cap.read() # Reads a frame
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    

    def play_video(self):
        for frame in self.frames:
            cv2.imshow("Video Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()


    def get_orignal(self):
        return self.__original


    def copy(self):
        dupe = c.deepcopy(self)
        return dupe
    
    # def undoable(func):
    #     def wrapper(self, inplace=False, *args, **kwargs):
    #         # hold_please = c.deepcopy(self)
    #         # output = func(self, *args, **kwargs)

    #         if inplace==True:
    #             self.__undo_list.append(c.deepcopy(self))
    #             if len(self.__undo_list) > self.UNDO_QUEUE_LENGTH:
    #                 self.__undo_list = self.__undo_list[1:]
    #         return func(self, *args, **kwargs)
    #     return wrapper
    
    # @undoable


    def for_all_frames(func):
        def wrapper(self, inplace=False, *args, **kwargs):
            out_frames = [func(self, frame, *args, **kwargs) for frame in self.frames]
            if inplace:
                self.frames = out_frames
                return self
            else:
                output:Trial = self.copy()
                output.frames = out_frames
                return output
        return wrapper
    

    ## TODO: Add an undo function
    # def undo(self, frame):

    def get_undo_length(self):
        out = len(self.__undo_list)
        print("Undo queue length:", out)
        return out
    

    # def undo(self):
    #     if len(self.__undo_list) < 1:
    #         raise IndexError("Undo queue is empty")
    #     print(self.__undo_list[0].shape)
    #     self.__copy_to(self.__undo_list.pop())
    #     print(self.shape)

    def __copy_to(self, other):
        self.__original = other.__original
        self.frames = other.frames
        self.shape = other.shape
        ## NOTE: This needs to be updated if other attributes are added

    # @undoable
    @for_all_frames
    def median_blur(self, frame, ksize=(3,3)):
        return cv2.medianBlur(frame, ksize=ksize)


    # @undoable
    @for_all_frames
    def to_gray(self, frame):
        if frame.ndim == 2:
            return frame
        self.shape = self.shape[:2]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @for_all_frames
    def gauss_blur(self, frame, ksize=(3,3), sigmaX=3, **kwargs):
        return cv2.GaussianBlur(frame, ksize, sigmaX, **kwargs)
    
    @for_all_frames
    def contour(self, frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE):
        return cv2.findContours(frame, mode, method)
    
    @for_all_frames
    def change_contrast(self, frame, alpha=1, beta=0):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    @for_all_frames
    def dilate(self, frame, kernel=cv2.MORPH_ELLIPSE, ksize=(3,3)):
        element = cv2.getStructuringElement(kernel, ksize)
        return cv2.dilate(frame, element)
    
    @for_all_frames
    def erode(self, frame, kernel=cv2.MORPH_ELLIPSE, ksize=(3,3)):
        element = cv2.getStructuringElement(kernel, ksize)
        return cv2.erode(frame, element)

    @for_all_frames
    def edges(self, frame, thresh1=25, thresh2=100):
        return cv2.Canny(frame, thresh1, thresh2)

    @for_all_frames
    def sharpen(self, frame, ddepth=-1):
        return cv2.filter2D(frame, ddepth=ddepth, kernel=self.sharpening_kernel)
    
    @for_all_frames
    def connect_le_components(self, frame, connectivity=8, ltype=cv2.CV_32S):
        _, out = cv2.connectedComponents(frame, connectivity=connectivity, ltype=ltype)
        return out
    
    @for_all_frames
    def crop_top(self, frame):
        y = frame.shape[0]
        y_min = (2 * y) // 5
        if frame.ndim == 2:
            new = frame[y_min:, :]
        else:
            new = frame[y_min:, :, :]
        self.shape = new.shape
        return new
    
    
    @for_all_frames
    def ranger(self, frame, minval, maxval):
        return cv2.inRange(frame, lowerb=minval, upperb=maxval)
    
    @for_all_frames
    def color_reduce(self, frame, K=4):
        Z = np.float32(frame.reshape((-1, 3)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(frame.shape)

        return quantized
    
    
    def remove_avg(self):

        img_avg = np.mean(self.frames, axis=0).astype('uint8')
        # img_avg = cv2.bitwise_not(img_avg)
        # self.show(img_avg)
        
        updated_frames = []
        for frame in self.frames:
            # frame = cv2.bitwise_not(frame)
            frame = frame - img_avg
            updated_frames.append(frame.astype('uint8'))

        self.frames = updated_frames

    def apply_mask(self, other, inplace=False):
        out = []
        for frame in zip(self.frames, other.frames):
            framey = frame[0]
            mask = frame[1]
            mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            # framey = *2//3 + framey//3
            framey = cv2.addWeighted(cv2.bitwise_and(framey, mask_3_channel), 2, framey, 0.5, 1)
            out.append(framey)
        
        if inplace:
            self.frames = out
        return out

        
        
    def create_overlay(self, other):
        other = self.__original
        L_out = []
        for frame in zip(other, self.frames):
            overlay = c.deepcopy(frame[0])  # .copy()
            overlay[frame[1] > 0] = [0, 255, 0]
            L_out.append(overlay)

        self.frames = L_out
        return L_out
    
    def show(self, frame):
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_average_img(self):
        out_img = self.frames[0]
        for i in range(len(self.frames[1:])):
            out_img = cv2.addWeighted(out_img*i/(i+1), 0, self.frames[i]/(i+1), 1, 0)
        
        return out_img


    def __getitem__(self, index):
        return self.frames[index]
    

    def __eq__(self, other) -> bool:
        if not isinstance(other, Trial):
            return TypeError
        if self.shape != other.shape:
            return False
        
        for self_frame, other_frame in zip(self.frames, other.frames):
            if self_frame.shape != other_frame.shape:
                return False
            if not (np.sum(self_frame==other_frame) == self_frame.size):
                return False
        
        for self_frame, other_frame in zip(self.__original, other.__original):
            if not (np.sum(self_frame==other_frame) == self_frame.size):
                return False
        
        return True
        

    



    