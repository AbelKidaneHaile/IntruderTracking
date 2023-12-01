# imports 
import cv2

# write the code to annotate a frame given a specified bounding boxes
def draw_annotation_mot(image, x1, y1, x2, y2, index):
    w = 10
    h = 10
    font_thickness = 5  
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 1.5
    color = (200, 0, 0)
    font_color = (200, 200, 200)
    # if index==16:
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 200), 2)
    #     color = (0, 0, 200)
    # else:
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 3)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 6)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 6)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 6)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 6)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 6)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 6)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 6)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 6)

    text = f'ID:{str(index)}'

    # Get the size of the text to determine the rectangle size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    rectangle_size = ((x1, y1 - text_height - 5), (x1 + text_width + 5, y1))
    # Draw the filled white rectangle
    cv2.rectangle(image, rectangle_size[0], rectangle_size[1], (255, 0, 255), cv2.FILLED)

    # if index==16: 
    #     cv2.putText(image, text,
    #                 (x1, y1 - 2),
    #                 0, 1.2, (0, 0, 255),
    #                 thickness=3, lineType=cv2.FILLED)
    # else:
    #     cv2.putText(image, text,
    #             (x1, y1 - 2),
    #             0, 1.2, (0, 255, 0),
    #             thickness=3, lineType=cv2.FILLED)

    # cv2.putText(image, text,
    #             (x1, y1 - 2),
    #             0, 1.2, (200, 0, 200),
    #             thickness=3, lineType=cv2.FILLED)

    cv2.putText(image, text, (x1, y1), font, font_scale, font_color, thickness=font_thickness)