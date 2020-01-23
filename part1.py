import moviepy.video.io.VideoFileClip as mpy
import cv2
import numpy as np
from moviepy.editor import *

walker = mpy.VideoFileClip("walker.avi")
walker_hand = mpy.VideoFileClip("walker_hand.avi")

frame_count = walker_hand.reader.nframes
video_fps = walker_hand.fps

kernel = np.ones((3,3),np.uint8)

frame_list = []

for t in range(1, frame_count-1):
    print(t, "\t", frame_count)
    walker_frame = walker.get_frame(t*1.0/video_fps)    #To get frames by ID
    gray_walker_frame = cv2.cvtColor(walker_frame, cv2.COLOR_BGR2GRAY)

    hand_frame = walker_hand.get_frame(t*1.0/video_fps)
    gray_hand_frame = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)

    erosion = cv2.erode(gray_hand_frame, kernel, iterations = 1)
    edges = gray_hand_frame - erosion

    edge_coords = np.where(edges > 200)

    next_frame = walker_hand.get_frame((t+1)*1.0/video_fps)    #To get frames by ID
    next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    Grad_t = (next_frame[edge_coords] - gray_hand_frame[edge_coords])

    Grad_x = (gray_hand_frame[(edge_coords[0]+1, edge_coords[1])] - gray_walker_frame[edge_coords])
    Grad_y = (gray_hand_frame[(edge_coords[0], edge_coords[1]+1)] - gray_walker_frame[edge_coords])

    Grad_t = Grad_t.flatten()   # Make it vector
    Grad_x = Grad_x.flatten()   # Make it vector
    Grad_y = Grad_y.flatten()   # Make it vector

    Grad_matrix = np.vstack((Grad_x, Grad_y)).T
    V = np.matmul(np.linalg.pinv(np.matmul(Grad_matrix.T, Grad_matrix)), np.matmul(Grad_matrix.T, -Grad_t))
    V = 40 * V / np.linalg.norm(V)   #make it unit vector and make it in leth of 40

    hand_points = np.where(gray_hand_frame > 127)   #The right hand
    point_coords = np.vstack((hand_points[0], hand_points[1])).T
    start_point = np.int16(np.mean(point_coords, axis=0))       #center of mass of the hand

    end_point = np.int16(start_point + V)
    start_point = (start_point[1], start_point[0])
    end_point = (end_point[1], end_point[0])

    walker_frame = cv2.arrowedLine(walker_frame, start_point, end_point, (255, 0, 0), 3)
    frame_list.append(walker_frame)

clip = ImageSequenceClip(frame_list, fps=(video_fps/2))
clip.write_videofile('OpticalFlow.mp4', codec='mpeg4')


"""
if t == 3:
    cv2.imshow("hand_frame", gray_hand_frame)
    cv2.imshow("edges", hand_edge_frame)
    erosion = cv2.erode(gray_hand_frame, kernel, iterations = 1)
    cv2.imshow("erosion", erosion)
    erosion_edges = cv2.Canny(erosion, 200, 250)
    cv2.imshow("erosion_edges", erosion_edges)
    fark = gray_hand_frame - erosion
    cv2.imshow("fark", fark)
    cv2.waitKey()
"""
