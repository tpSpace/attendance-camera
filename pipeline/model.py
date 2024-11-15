from typing import Iterator, List, Optional, OrderedDict, Tuple, Union
import cv2
import dlib
import numpy as np


class Person:
    COUNTER = 0
    def __init__(self, name:str = None):
        if name is None:
            name = f"Person_{Person.COUNTER}"
            Person.COUNTER += 1
        else:
            name = name
    
    def set_landmark(self, landmark: Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray], Optional[List[np.ndarray]]]]):
        self.landmark = landmark

    def set_embedding(self, embedding: np.ndarray):
        self.embedding = embedding
    
class BoundingBox:
    """
    Represents a bounding box in an image.
    
    A bounding box is defined by its top-left and bottom-right corners (x1, y1, x2, y2) and a confidence score.
    
    Attributes:
        x1 (int): x-coordinate of the top-left corner.
        y1 (int): y-coordinate of the top-left corner.
        x2 (int): x-coordinate of the bottom-right corner.
        y2 (int): y-coordinate of the bottom-right corner.
        confidence (float): confidence score of the bounding box.
    """
    def __init__(self, x1: int, y1: int, x2: int, y2: int, confidence: float):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, float]]:
        return iter((self.x1, self.y1, self.x2, self.y2, self.confidence))

    def tl(self) -> Tuple[int, int]:
        """
        Returns the top-left corner of the bounding box.
        """
        return self.x1, self.y1
    
    def br(self) -> Tuple[int, int]:
        """
        Returns the bottom-right corner of the bounding box.
        """
        return self.x2, self.y2
    
    def area(self) -> int:
        """
        Returns the area of the bounding box.
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def height(self) -> int:
        """
        Returns the height of the bounding box.
        """
        return self.y2 - self.y1
    
    def width(self) -> int:
        """
        Returns the width of the bounding box.
        """
        return self.x2 - self.x1
    
    def to_dlib_rect(self):
        """
        Returns a dlib rectangle object from the bounding box.
        """
        return dlib.rectangle(left=self.x1, top=self.y1, right=self.x2, bottom=self.y2)
    
    def to_list(self) -> List[int]:
        """
        Returns the bounding box as a list.
        """
        return [self.x1, self.y1, self.x2, self.y2, self.confidence]
    
    def crop(self, image: np.ndarray) -> np.ndarray:
        """
        Crop the bounding box from the image.
        
        Arguments:
            image {numpy.ndarray} -- The input image.
        
        Returns:
            numpy.ndarray -- The cropped image.
        """
        return image[self.y1:self.y2, self.x1:self.x2]
    

    