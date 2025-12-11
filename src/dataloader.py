import cv2
import numpy as np
import copy as c

class MOVE:
    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) 


    def __init__(self, video_path:str):
        self.frames = self.load_mov_as_array(video_path)
        # self.__original = self.frames
        self.shape = self.frames[0].shape

    @classmethod
    def saveImg(cls, img, name:str):
        cv2.imwrite(f"../saved_imgs/{name}.png", img)

    def saveFrames(self, name:str):
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(f"{name}.mov", fourcc, 30, (self.frames[0].shape[1],self.frames[0].shape[0]))
        for frame in self.frames:
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.release()

        
    @classmethod
    def show(self, frame):
        """Shows the inputted image"""
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def load_mov_as_array(self, video_path):
        """Loads the frames of a .mov file into the MOVE object given its filepath/name"""

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
        """Plays the processed MOVE recording"""

        for frame in self.frames:
            cv2.imshow("Video Frame", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    
    def create_average_img(self):
        """Generates the time-averaged background image of self.frames"""
        out_img = self.frames[0]
        for i in range(len(self.frames[1:])):
            out_img = cv2.addWeighted(out_img*i/(i+1), 0, self.frames[i]/(i+1), 1, 0)
        
        return out_img


    def copy(self):
        """Makes a deep copy of the object"""
        dupe = c.deepcopy(self)
        return dupe
    

    def for_all_frames(func): 
        def wrapper(self, inplace=False, *args, **kwargs):
            out_frames = [func(self, frame, *args, **kwargs) for frame in self.frames]
            if inplace:
                self.frames = out_frames
                return self
            else:
                output:MOVE = self.copy()
                output.frames = out_frames
                return output
        return wrapper
    

    # -----------------------
    # COLOR FILTERS
    # -----------------------

    @for_all_frames  
    def to_gray(self, frame):
        """Convert image to grayscale"""
        if frame.ndim == 2:
            return frame
        self.shape = self.shape[:2]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    @for_all_frames
    def ranger(self, frame, minval, maxval):
        return cv2.inRange(frame, lowerb=minval, upperb=maxval)
    
    @for_all_frames 
    def color_reduce(self, frame, K=4):
        """DO NOT USE. FAR TOO COMPUTATIONALLY EXPENSIVE"""
        Z = np.float32(frame.reshape((-1, 3)))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        quantized = centers[labels.flatten()]
        quantized = quantized.reshape(frame.shape)

        return quantized
    

    # ---------------------------------
    #  BLURRING FILTERS
    # ---------------------------------
    
    @for_all_frames  
    def gauss_blur(self, frame, ksize=(3,3), sigmaX=3, **kwargs):
        return cv2.GaussianBlur(frame, ksize, sigmaX, **kwargs)
    
    @for_all_frames   
    def median_blur(self, frame, ksize=3, **kwargs):
        
        return cv2.medianBlur(frame, ksize=ksize, **kwargs)
    
    @for_all_frames  
    def sharpen(self, frame, ddepth=-1, **kwargs):
        return cv2.filter2D(frame, ddepth=ddepth, kernel=self.sharpening_kernel, **kwargs)
    
    @for_all_frames
    def edge_blur(self, frame, **kwargs):
        return cv2.edgePreservingFilter(frame, **kwargs)
    

    # ---------------------------------
    #  MORPHOLOGICAL TRANSFORMATIONS
    # ---------------------------------

    @for_all_frames
    def change_contrast(self, frame, alpha=1, beta=0, **kwargs):
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta, **kwargs)

    @for_all_frames  
    def dilate(self, frame, kernel=cv2.MORPH_ELLIPSE, ksize=(3,3), **kwargs):
        element = cv2.getStructuringElement(kernel, ksize)
        return cv2.dilate(frame, element, **kwargs)
    
    @for_all_frames  
    def erode(self, frame, kernel=cv2.MORPH_ELLIPSE, ksize=(3,3), **kwargs):
        element = cv2.getStructuringElement(kernel, ksize)
        return cv2.erode(frame, element, **kwargs)
    
    @for_all_frames
    def opening(self, frame, ksize=(3,3), **kwargs):
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, ksize, **kwargs)
    
    @for_all_frames
    def closing(self, frame, ksize=(3,3), iterations=1, **kwargs):
        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, ksize, iterations=iterations, **kwargs)


    # ---------------------------------
    #  FEATURE DETECTION
    # ---------------------------------

    @for_all_frames  
    def edges(self, frame, thresh1=25, thresh2=100, **kwargs):
        return cv2.Canny(frame, thresh1, thresh2, **kwargs)
    
    @for_all_frames  
    def contour(self, frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, **kwargs):
        return cv2.findContours(frame, mode, method, **kwargs)
    
    @for_all_frames  
    def connect_le_components(self, frame, connectivity=8, ltype=cv2.CV_32S, **kwargs):
        _, out = cv2.connectedComponents(frame, connectivity=connectivity, ltype=ltype, **kwargs)
        return out
    
    @for_all_frames
    def img_like(self, frame, img):
        """Replaces each frame with img"""
        return img
    

    # ---------------------------------
    #  Masking Utilities
    # ---------------------------------

    def get_avg(self, skipframes=0) -> np.ndarray:
        """
        Creates the time-average image to act as a pseudo-background 
        (static) image

        :param skipframes: Optional parameter to dictate how many frames 
            to skip from the start to prevent over-representation of the
            starting fin position in the pseudo-background image
        """
        return np.mean(self.frames[skipframes:], axis=0).astype('uint8')
    
    
    def remove_avg(self, skipframes=0, happy_accident_mode=True):
        """
        Subtracts the average image from all frames

        :param skipframes: Optional parameter to dictate how many frames 
            to skip from the start to prevent over-representation of the
            starting fin position in the pseudo-background image
        :param happy_accident_mode: When False, this mode takes the absolute
            difference between the two images it compares. When enabled, the
            function merely subtracts the average image from a given image,
            meaning that when the average image is brighter than the given
            image, the pixel's value becomes negative which wraps around to
            positive, making these spots extremely bright. This had an
            unintentionally positive effect that greatly improved the 
            pipeline's results
        """

        img_avg = self.get_avg(skipframes=skipframes)
        
        updated_frames = []
        for frame in self.frames:
            if happy_accident_mode:
                frame = frame - img_avg 
            else:
                frame = cv2.absdiff(frame, img_avg)

            updated_frames.append(frame.astype('uint8'))

        self.frames = updated_frames


    def apply_mask(self, other, inplace=False):
        """
        Overlays a mask video that highlights the masked regions of each frame
            and darkens unmasked regions
        
        :param other: The 
        :param inplace: Description
        """
        out = []
        for frame in zip(self.frames, other.frames):
            framey = frame[0]
            mask = frame[1]
            mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            framey = cv2.addWeighted(cv2.bitwise_and(framey, mask_3_channel), 2, framey, 0.5, 1)
            out.append(framey)
        
        if inplace:
            self.frames = out
        return out
    

    def combine_masks(self, other, inplace=False):
        """Combines two masks"""
        out = []
        for frame in zip(self.frames, other.frames):
            frameS, frameO = frame
            maskS = np.bitwise_and(frameS, np.ones_like(frameS))
            maskO = np.bitwise_and(frameO, np.ones_like(frameS))
            multi_mask = (127*np.where(maskS==1, 2*maskS, maskO)).astype('uint8')
            out.append(multi_mask)

        if inplace:
            self.frames = out
        return out
    
    @for_all_frames
    def generic_filter(self, frame, /, func, **kwargs):
        return func(frame, **kwargs)

    def __getitem__(self, index):
        img = self.frames[index]
        return img
    

    def __eq__(self, other) -> bool:
        if not isinstance(other, MOVE):
            return TypeError
        if self.shape != other.shape:
            return False
        
        for self_frame, other_frame in zip(self.frames, other.frames):
            if self_frame.shape != other_frame.shape:
                return False
            if not (np.sum(self_frame==other_frame) == self_frame.size):
                return False
        return True
        

MOVE.median_blur.__doc__ = cv2.medianBlur.__doc__
MOVE.sharpen.__doc__ = cv2.filter2D.__doc__
MOVE.edge_blur.__doc__ = cv2.edgePreservingFilter.__doc__
MOVE.change_contrast.__doc__ = cv2.convertScaleAbs.__doc__
MOVE.edges.__doc__ = cv2.Canny.__doc__
MOVE.contour.__doc__ = cv2.findContours.__doc__
MOVE.connect_le_components.__doc__ = cv2.connectedComponents.__doc__


    