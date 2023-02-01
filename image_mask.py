from imutils.video import FileVideoStream
import numpy as np
import argparse
import cv2
import math

width_adjust1 = 15
width_adjust2 = 25
height_adjust = 0

def find_param(camera_height, camera_distance, camera_vertical_fov , image_pix_height = 768 ):
    number = camera_distance/camera_height
    angle = round(math.degrees(math.atan(number)))
    print("camera_distance :", camera_distance)
    print("camera_height :", camera_height)
    print("angle:",angle)
    angle1 = round(camera_vertical_fov + angle)
    print("angle1:",angle1)
    #print("tan of angle1 :", math.tan(math.radians(angle1)))
    #print("tan of angle :",  math.tan(math.radians(angle)))
    change_param_sum = 0
    for i in range(angle, angle1 , 1) :
        param1 = ( math.tan( math.radians(i + 1) ) - math.tan( math.radians(i) ) ) * ( math.cos( math.radians(i + 1) ) ) / (math.cos ( math.radians(i) ) )
        #print("param1: ",i, param1)
        param2 = math.tan(math.radians(i + 2)) - math.tan(math.radians(i + 1))* ( math.cos (math.radians(i + 2) ) ) / ( math.cos( math.radians(i + 1) ) )
        #print("param2: ",i, param2)
        change_param = param2 - param1
        change_param_sum = change_param_sum + change_param
    dist_change_param = change_param_sum/(angle1 - angle)
    pix_to_dist_change_param = dist_change_param * camera_vertical_fov / image_pix_height 
    
    return pix_to_dist_change_param



def find_fov_length(camera_height,camera_distance,camera_vertical_fov):
    number = camera_distance/camera_height
    angle = math.degrees(math.atan(number))
    print("camera_distance :", camera_distance)
    print("camera_height :", camera_height)
    print("angle:",angle)
    angle1 = camera_vertical_fov + angle
    print("angle1:",angle1)
    print("tan of angle1 :", math.tan(math.radians(angle1)))
    print("tan of angle :",  math.tan(math.radians(angle)))
    
    fov_length = (math.tan(math.radians(angle1)) - math.tan(math.radians(angle)) )*camera_height
    #pix_to_dist_change_param = 0.0002
    return fov_length
   


def calculate_per_pixel_distance(pix_to_dist_change_param,fov_length,image_pix_height = 768):
    a = pix_to_dist_change_param#0.0004
    b = 2*image_pix_height#1536
    c = -2*fov_length#-28.292

    # calculate the discriminant
    d = (b**2) - (4*a*c)

    # find two solutions
    sol1 = (-b-math.sqrt(d))/(2*a)
    sol2 = (-b+math.sqrt(d))/(2*a)
    pix_to_dist_param = max(sol1,sol2)
    return pix_to_dist_param

 
'''
def calculate_per_pixel_distance(pix_to_dist_change_param,fov_length,image_pix_height = 768):
    #pix_to_dist_change_param = 0.0002
    added_distance = pix_to_dist_change_param * (image_pix_height*(image_pix_height - 1)/2)
    print("added_distance :", added_distance)
    pix_to_dist_param = (fov_length - added_distance)/768
    return pix_to_dist_param
'''
def get_bloom_length(starting_pix , updated_length_of_bloom_in_pix , pix_to_dist_change_param , updated_pix_to_dist_change_param , pix_to_dist_param):
    #pix_to_dist_change_param = 0.0002
    starting_pix_dist = (pix_to_dist_param * starting_pix  + pix_to_dist_param * (pix_to_dist_param - 1) *  pix_to_dist_change_param / 2) / starting_pix
    length_of_bloom = updated_length_of_bloom_in_pix * starting_pix_dist + updated_pix_to_dist_change_param * pix_to_dist_param * (pix_to_dist_param - 1) / 2
    return length_of_bloom


camera_height = math.sqrt(15.4**2 + 2**2)

camera_distance = 3

camera_vertical_fov = 36.9 


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", type=str, default = "/home/dss/Downloads/bar_rod/image/1.png" , help ="path to the input image" )
args = vars(ap.parse_args())
image = cv2.imread(args["image"])


'''
print("[INFO] starting video stream...")
vs = FileVideoStream(path='./bloom/bloom1.mp4').start()
'''

cv2.imshow("Original", image)
#image1 = vs.read()
#image = cv2.resize(image1, (1366,768), interpolation = cv2.INTER_AREA)
mask = np.zeros(image.shape[:2], dtype="uint8")
#cv2.polygon(mask, (670, 10), (850,10) , (1270,750) ,  (950, 750), 255, -1)
#pts = np.array([[670,10], [830,10], [1270,750], [950,750]])
pts = np.array([[0,340], [1920,340], [1920,1080], [0,1080]])
isClosed = True
color = 255
thickness = -1
  
#polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> img
cv2.fillPoly(mask, [pts], color)
cv2.imshow("Rectangular Mask", mask)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask Applied to Image", masked)
cv2.waitKey(0)
#imageGray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray_img",imageGray)
(T, imagethresh) = cv2.threshold(imageGray, 100, 255,cv2.THRESH_BINARY_INV)
cv2.imshow("thresh_img",imagethresh)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(imagethresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  
cv2.imshow('apply Contours', imagethresh)
cv2.waitKey(0)
  
print("Number of Contours found = " + str(len(contours)))
  
# Draw all contours
# -1 signifies drawing all contours
#cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
  
cv2.imshow('Contours', image)
cv2.waitKey(0)
idx = 0
for cnt in contours:
     idx += 1
     x,y,w,h = cv2.boundingRect(cnt)
     roi=image[y:y+h,x:x+w]
     #if w < 500 and h > 600 :
     if w < 1000 and h > 200 :
        cv2.drawContours(image, cnt, -1, (0, 255, 0), 3)
        print("x:",x)
        print("y:",y) 
        print("width:",w)
        print("height:",h)
        cv2.imwrite(str(idx) + '.jpg', roi)
        '''
        updated_height_in_pix = h - height_adjust

        print("updated_height_in_pix :", updated_height_in_pix)

        updated_width_in_pix = w - width_adjust1 - width_adjust2

        print("updated_width_in_pix :", updated_width_in_pix)

        updated_x_in_pix = x + width_adjust1

        print("updated_x_in_pix :", updated_x_in_pix)

        updated_length_of_bloom_in_pix = math.sqrt(updated_height_in_pix**2 + updated_width_in_pix**2)

        print("updated_length_of_bloom_in_pix :", updated_length_of_bloom_in_pix)

        starting_pix = 768 - y - updated_height_in_pix 

        print("starting_pix :", starting_pix)

        coeff_of_angle_of_bloom = updated_length_of_bloom_in_pix/updated_height_in_pix

        print("coeff_of_angle_of_bloom :", coeff_of_angle_of_bloom)

        pix_to_dist_change_param = find_param(camera_height,camera_distance,camera_vertical_fov)

        print("pix_to_dist_change_param :", pix_to_dist_change_param)

        fov_length = find_fov_length(camera_height,camera_distance,camera_vertical_fov)
       
        print("fov_length :", fov_length)

        pix_to_dist_param = calculate_per_pixel_distance(pix_to_dist_change_param,fov_length)

        print("pix_to_dist_param :", pix_to_dist_param)


        updated_pix_to_dist_change_param = pix_to_dist_change_param / coeff_of_angle_of_bloom

        print("updated_pix_to_dist_change_param :", updated_pix_to_dist_change_param)

        length_of_bloom = get_bloom_length(starting_pix,updated_length_of_bloom_in_pix,pix_to_dist_change_param,updated_pix_to_dist_change_param,pix_to_dist_param)  

        print("length_of_bloom :", length_of_bloom)
        '''
     cv2.imshow("final",image)
     key = cv2.waitKey(1) & 0xFF
     #count1 = count1 + 1
     # if the `q` key was pressed, break from the loop
     if key == ord("q"):
        break 
cv2.destroyAllWindows()
