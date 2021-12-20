import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


# Step 1: Basic Setting
# 보여지는 이미지들의 배치 스타일
def plot_img(rows, cols, index, img, title):
    ax = plt.subplot(rows,cols,index)
    if(len(img.shape) == 3):
        ax_img = plt.imshow(img[...,::-1])
    else:
        ax_img = plt.imshow(img, cmap='gray')
    plt.axis('on')
    if(title != None): plt.title(title) 
    return ax_img, ax

# 변형되지 않는 기준 이미지, 변형될 이미지, 마스크 이미지 배열
criteria_imgs, transform_imgs, masks = [], [], [] 

# 단계별로 수행되는 마스크 추출에 사용되는 이미지들
result1s, result2s = [], []

#: 기준이 될 이미지, 변형시킬 이미지 경우에 따라 삽입
for i in range(5):
    img = cv.imread(str(i+1)+'.jpeg')
    img = cv.resize(img, (500,500))
    if i != 0:
        transform_imgs.append(img)
    else:
        i = np.zeros((img.shape[0]*3, img.shape[1],3), np.uint8)
        i[img.shape[0]:img.shape[0]*2, :] = img
        criteria_imgs.append(i)
        
        
for i in range(4):
    # Step 2: 단계별 이미지 스티칭
    # c_XXX... : about criteria images
    # t_XXX... : about transform images
    
    # 스티칭 작업에 이용될 변수들 Setting
    # c,t에 대한 각각의 grayscale image / kp & des / gray scale kfp / matches
    c_img = criteria_imgs[i]
    t_img = transform_imgs[i]
        
    c_img_gray = cv.cvtColor(c_img, cv.COLOR_BGR2GRAY)
    t_img_gray = cv.cvtColor(t_img, cv.COLOR_BGR2GRAY)
        
    sift = cv.SIFT_create()
    c_kp, c_des = sift.detectAndCompute(c_img, None)
    t_kp, t_des = sift.detectAndCompute(t_img, None)
        
    c_img_gray_kfp = cv.drawKeypoints(c_img_gray, c_kp, 
                                      None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    t_img_gray_kfp = cv.drawKeypoints(t_img_gray, t_kp, 
                                      None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    flann = cv.FlannBasedMatcher({"algorithm":1, "trees":5}, {"checks":50})
    matches = flann.knnMatch(c_des, t_des, k=2)
    
    # Coreespondences 수치 저장
    # 매칭되는 점끼리 초록색 선으로 연결하여 보여줌 (시각화)
    good_correspondences = []
    def update_good_correpondences(ratio_dist):
        good_correspondences.clear()
        for m,n in matches:
            if m.distance/n.distance < ratio_dist:
                good_correspondences.append(m)
    
    update_good_correpondences(0.7)
    img_matches = cv.drawMatches(c_img,c_kp,t_img,t_kp,good_correspondences,None,
                                 matchColor=(0,255,0),singlePointColor=None,matchesMask=None,flags=2)
    fig = plt.figure(1)
    fig.canvas.mpl_connect('close_event', lambda e : plt.close('all'))
    ax_img, ax = plot_img(1,4,i+1,img_matches,"For Matching Step "+str(i+1))

    # correspondence 수치 보여줌
    tx = ax.text(0.05, 0.95, "# good correspondences: " +
                 str(len(good_correspondences)), transform=ax.transAxes, fontsize=7,
            verticalalignment='top', bbox={'boxstyle':'round', 'facecolor':'wheat', 'alpha':0.5})
    tx.set_text("good correspondences "+str(i+1)+': ' + str(len(good_correspondences)))  

    # Stitching 단계를 보여주는 버튼 생성
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Stitch', color='lightgoldenrodyellow', hovercolor='0.975')
    
    # Correspondences를 통한 H값 도출 (매트릭스 도출)
    src_pts = np.float32([ c_kp[m.queryIdx].pt for m in good_correspondences ]).reshape(-1,1,2)
    dst_pts = np.float32([ t_kp[m.trainIdx].pt for m in good_correspondences ]).reshape(-1,1,2)
    H, mask = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 5.0)
    
    # 스티칭할 이미지들 크기 지정
    stitch_plane_rows = t_img.shape[0]*3
    stitch_plane_cols = t_img.shape[1] + c_img.shape[1] # 반복할 때마다 가로 길이가 늘어남
    
    # H값(기준 이미지)에 따라 변형될 이미지 변형
    result1 = cv.warpPerspective(t_img, H, (stitch_plane_cols, stitch_plane_rows),
                                 flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
    
    # 지정된 크기의 이미지에 기준 이미지 삽입
    result2 = np.zeros((stitch_plane_rows, stitch_plane_cols, 3), np.uint8)
    result2[0:c_img.shape[0], 0:c_img.shape[1]] = c_img
    
    # result1, result2와 서로 겹치는 마스크 도출
    and_img = cv.bitwise_and(result1, result2)
    and_img_gray = cv.cvtColor(and_img, cv.COLOR_BGR2GRAY)
    th, mask = cv.threshold(and_img_gray, 1, 255, cv.THRESH_BINARY)
    
    # 결과 이미지 저장
    result3 = np.zeros((stitch_plane_rows, stitch_plane_cols,3), np.uint8)
    result2_gray = cv.cvtColor(result2, cv.COLOR_BGR2GRAY)
    for y in range(stitch_plane_rows):
        for x in range(stitch_plane_cols):
            mask_v1 = mask[y, x]
            if(mask_v1 > 0): # 마스크 내부에는 result1, result2이미지 반반씩 섞어 저장
                result3[y, x] = np.uint8(result1[y,x] * 0.5 + result2[y,x] * 0.5)
            elif(mask_v1 == 0 and result2_gray[y,x] == 0): # 마스크 외부 오른쪽에 result1이미지 삽입
                result3[y, x] = result1[y,x]
            else: # 마스크 외부 왼쪽에 result2 이미지 삽입
                result3[y, x] = result2[y,x]
                          
    result1s.append(result1)
    result2s.append(result2)
    masks.append(mask)
    criteria_imgs.append(result3)      
                    

# Step3: 스티치 과정을 보여줄 함수
def stitch(event):
    for i in range(4):
        plt.figure(2)
        plot_img(4, 3, 3*i+1, criteria_imgs[i], None)
        plot_img(4, 3, 3*i+2, result1s[i], None)
        plot_img(4, 3, 3*i+3, masks[i], None)
            
    plt.figure(3)
    plot_img(1, 1, 1, criteria_imgs[4], None) # 스티칭 최종 결과
    plt.show()
    

button.on_clicked(stitch)
plt.show() 
