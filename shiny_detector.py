import cv2
import numpy as np

#desired_pokemon = 'normal_pokemon/483.png'
#desired_pokemon = 'normal_pokemon/483.png'

target_img = cv2.imread('399.png')
screenshot_img = cv2.imread('screenshot.png')

#cv2.imshow('Dialga', target_img)
#cv2.waitKey()
#cv2.destroyAllWindows()

#cv2.imshow('Screen', screenshot_img)
#cv2.waitKey()
#cv2.destroyAllWindows()

result = cv2.matchTemplate(screenshot_img, target_img, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(max_loc)
print(max_val)

w = target_img.shape[1]
h = target_img.shape[0]

cv2.rectangle(screenshot_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), 2)

cv2.imshow('Screen', screenshot_img)
cv2.waitKey()
cv2.destroyAllWindows()

#threshold = 0.8

#loc = np.where( result >= threshold)

#for pt in zip(*loc[::-1]):
#    cv2.rectangle(screenshot_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

#cv2.imwrite('res.png', screenshot_image)

#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#top_left = max_loc

#bottom_right = (top_left[0] + w, top_left[1] + h)

#cv2.rectangle(screenshot_image, top_left, bottom_right, 255, 2)
