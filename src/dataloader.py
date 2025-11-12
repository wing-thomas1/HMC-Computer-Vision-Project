import cv2
import numpy as np


class Trial:
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    def __init__(self, video_path):
        if isinstance(video_path, str):
            self.frames = self.load_mov_as_array(video_path)
        elif isinstance(video_path, list):
            self.frames = video_path
        else:
            TypeError("Not a valid type")
        

        self.__original = self.frames

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
    
    
    def for_all_frames(func):
        def wrapper(self, inplace=False, *args, **kwargs):
            out_frames = [func(self, frame, *args, **kwargs) for frame in self.frames]

            if inplace:
                self.frames = out_frames
                return self
            
            return Trial(out_frames)
        
        return wrapper
    

    ## TODO: Add an undo function
    # def undo(self, frame):

    @for_all_frames
    def median_blur(self, frame, ksize=(3,3)):
        return cv2.medianBlur(frame, ksize=ksize)


    @for_all_frames
    def to_gray(self, frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @for_all_frames
    def gauss_blur(self, frame, ksize=(3,3), sigmaX=3, **kwargs):
        return cv2.GaussianBlur(frame, ksize, sigmaX, **kwargs)
    
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
            return frame[y_min:, :]
        else:
            return frame[y_min:, :, :]
        
        
    def create_overlay(self, other):
        L_out = []
        # if other[0].shape[:3] != self.frames[0].shape[:3]:
        #     print("No can do, buckaroo")
        #     return None
        for frame in zip(other, self.frames):
            overlay = frame[0].copy()
            # print("Frame[1].shape:", frame[1].shape)
            # print("Frame[0].shape:", frame[0].shape)
            overlay[frame[1] > 0] = [0, 255, 0]
            L_out.append(overlay)
            
        return Trial(L_out)
    
    # def create_average_img(self):
    #     out_img = self.__original[0]
    #     for i in self.__original[1:]:
    #         out_img += i
        
    #     return out_img / (len(self.__original) - 1)


    def create_average_img(self):
        out_img = self.frames[0]
        for i in range(len(self.frames[1:])):
            out_img = cv2.addWeighted(out_img*i/(i+1), 0, self.frames[i]/(i+1), 1, 0)
        
        return out_img


    def __getitem__(self, index):
        return self.frames[index]


    