import cv2
import numpy as np
from matplotlib import pyplot

#desired_pokemon = 'normal_pokemon/483.png'
desired_pokemon = 'normal_pokemon/483.png'

target_pokemon = cv2.imread(desired_pokemon, cv2.IMREAD_UNCHANGED)
screenshot_image = cv2.imread('screenshot.png')
screenshot_image_gray = cv2.cvtColor(screenshot_image, cv2.COLOR_BGR2GRAY)
w, h = target_pokemon.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(screenshot_image_gray, target_pokemon, cv2.TM_CCOEFF_NORMED)

threshold = 0.8

loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(screenshot_image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imwrite('res.png', screenshot_image)

#min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

#top_left = max_loc

#bottom_right = (top_left[0] + w, top_left[1] + h)

#cv2.rectangle(screenshot_image, top_left, bottom_right, 255, 2)

#pyplot.subplot(121), pyplot.imshow(res,cmap = 'gray')
#pyplot.title('Matching Result'), pyplot.xticks([]), pyplot.yticks([])

#pyplot.subplot(122), pyplot.imshow(res,cmap = 'gray')
#pyplot.title('Detect Result'), pyplot.xticks([]), pyplot.yticks([])
#pyplot.suptitle('Result')

#pyplot.show()