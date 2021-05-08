from __future__ import division
import cv2
import Image, numpy as np,math
import scipy as sp
from numpy.linalg import eig, inv
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from pylab import *
from skimage import io, color
from scipy import interpolate


#Reqd color
R,G,B = (200.,121.,46.)
R,G,B = (234.,135.,103.)

#Intensity low = 0.5, med = 0.6, high = 0.7
inten=0.6

up_left_end = 3
up_right_end = 5

eye_lower_left_end = 5
eye_upper_left_end = 10
eye_lower_right_end = 15
eye_upper_right_end = 20


def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a < c:
            return 0
        else:
            return np.pi/2
    else:
        if a < c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

def getEllipse(x,y):
    arc = 0.8
    R = np.arange(0,arc*np.pi, 0.01)
    a = fitEllipse(x,y)
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    return (center[0],center[1]),(axes[0],axes[1]),phi

def univariate_plot(lx=[],ly=[]):
    unew = np.arange(lx[0], lx[-1]+1, 1)
    f2 = InterpolatedUnivariateSpline(lx, ly)
    return unew,f2(unew)

def inter_plot(lx=[],ly=[],k1='quadratic'):
    unew = np.arange(lx[0], lx[-1]+1, 1)
    f2 = interp1d(lx, ly, kind=k1)
    return unew,f2(unew)

def getBoundaryPoints(x , y):
    tck,u = interpolate.splprep([x, y], s=0, per=1)
    unew = np.linspace(u.min(), u.max(), 10000)
    xnew,ynew = interpolate.splev(unew, tck, der=0)
    tup = c_[xnew.astype(int),ynew.astype(int)].tolist()
    coord = list(set(tuple(map(tuple, tup))))
    coord = np.array([list(elem) for elem in coord])
    return coord[:,0],coord[:,1]

def getInteriorPoints(x , y):
    intx = []
    inty = []
    def ext(a, b, i):
     a, b=round(a), round(b)
     intx.extend(arange(a, b, 1).tolist())
     inty.extend((ones(b-a)*i).tolist())
    x, y = np.array(x), np.array(y)
    xmin, xmax = amin(x), amax(x)
    xrang = np.arange(xmin, xmax+1, 1)
    for i in xrang:
     ylist = y[where(x==i)]
     ext(amin(ylist), amax(ylist), i)
    return intx, inty

def checkForSkin(IMG10):
    high,widt=IMG10.shape[:2]

    B1=np.reshape(np.float32(IMG10[:,:,0]),high*widt)#B
    G1=np.reshape(np.float32(IMG10[:,:,1]),high*widt)#G
    R1=np.reshape(np.float32(IMG10[:,:,2]),high*widt)#Rs

    #print high,widt
    h3=np.zeros((high,widt,3),np.uint8)

    #cv2.imshow("onetime",h)
    

    tem=np.logical_and(np.logical_and(np.logical_and(np.logical_and(R1 > 95, G1 > 40),np.logical_and(B1 > 20, (np.maximum(np.maximum(R1,G1),B1) - np.minimum(np.minimum(R1,G1),B1)) > 15)),R1>B1),np.logical_and(np.absolute(R1-G1) > 15,R1>G1))
    h5=np.array(tem).astype(np.uint8,order='C',casting='unsafe')

    h5=np.reshape(h5,(high,widt))
    h3[:,:,0]=h5
    h3[:,:,1]=h5
    h3[:,:,2]=h5
    #cv2.imshow("thirdtime",h3)
    kernel1 = np.ones((3,3),np.uint8)
    closedH3=np.copy(h3)
    for i in range(5):
        closedH3 = cv2.erode(closedH3,kernel1)
    for i in range(5):
        closedH3 = cv2.dilate(closedH3,kernel1)
    #cv2.imshow("closedH3",closedH3)
    # closedH3 = cv2.cvtColor(closedH3, cv2.COLOR_BGR2RGB)
    return closedH3

fileface = np.loadtxt('pointface.txt')
pointsface =  np.floor(fileface)
point_face_x = np.array((pointsface[:][:,0]))
point_face_y = np.array(pointsface[:][:,1])

file = np.loadtxt('pointlips.txt')
points =  np.floor(file)
point_out_x = np.array((points[:len(points)/2][:,0]))
point_out_y = np.array(points[:len(points)/2][:,1])
point_in_x = (points[len(points)/2:][:,0])
point_in_y = points[len(points)/2:][:,1]

fileeye = np.loadtxt('pointeyes.txt')
pointseye =  np.floor(fileeye)
eye_point_down_x = np.array((pointseye[:eye_lower_left_end][:,0]))
eye_point_down_y = np.array(pointseye[:eye_lower_left_end][:,1])
eye_point_up_x = np.array(pointseye[eye_lower_left_end:eye_upper_left_end][:,0])
eye_point_up_y = np.array(pointseye[eye_lower_left_end:eye_upper_left_end][:,1])
eye_point_down_x_right = np.array((pointseye[eye_upper_left_end:eye_lower_right_end][:,0]))
eye_point_down_y_right = np.array(pointseye[eye_upper_left_end:eye_lower_right_end][:,1])
eye_point_up_x_right = np.array((pointseye[eye_lower_right_end:eye_upper_right_end][:,0]))
eye_point_up_y_right = np.array(pointseye[eye_lower_right_end:eye_upper_right_end][:,1])

im = imread('Input.jpg')
im2 = im.copy()
height, width = im.shape[:2]

x_face = []
y_face = []
x_aux = []
y_aux = []

# Face
lower_face = univariate_plot(point_face_x[:],point_face_y[:])
x_face.extend(lower_face[0][::-1])
y_face.extend(lower_face[1][::-1])

# Upper face approximation
(centerx,centery),(axesx,axesy),angel = getEllipse(point_face_x,point_face_y)
centerpt = (int(centerx),int(centery))
axeslen = (int(axesx),int(axesy*1.2))
# cv2.ellipse(im,centerpt,axeslen,angel,180,360,(0,255,0),2)
# upper_ellipse = cv2.ellipse2Poly(centerpt,axeslen,int(angel),180,360,1)
# upperellipse[1] = cv2.ellipse2Poly(centerpt,axeslen,int(angel),180,360,1)[:,1]
ellippoints = cv2.ellipse2Poly(centerpt,axeslen,int(angel),180,360,1)
ellippoints = np.floor(ellippoints)
ellipseabs = ellippoints[:,0].tolist()
ellipseord = ellippoints[:,1].tolist()
# upper_ellipse = univariate_plot(ellipseabs, ellipseord)
# upper_ellipse = inter_plot(ellipseabs, ellipseord, 'linear')
x_face.extend(ellipseabs)
y_face.extend(ellipseord)

x_face.append(x_face[0])
y_face.append(y_face[0])

x_face, y_face = getBoundaryPoints(x_face, y_face)
# print upper_ellipse[0], upper_ellipse[1]

# imshow(im)
# # plot(upper_ellipse[0], upper_ellipse[1], 'g-')
# plot(x_face, y_face, 'go')
# gca().set_aspect('equal', adjustable='box')
# imsave('out1.jpg',im)
# show()
x, y = getInteriorPoints(x_face, y_face)

#Lips
l_u_l = inter_plot(point_out_x[:up_left_end],point_out_y[:up_left_end])
l_u_r = inter_plot(point_out_x[up_left_end-1:up_right_end],point_out_y[up_left_end-1:up_right_end])
l_l = inter_plot([point_out_x[0]]+point_out_x[up_right_end-1:][::-1].tolist(),[point_out_y[0]]+point_out_y[up_right_end-1:][::-1].tolist(),'cubic')
lipinteriorx, lipinteriory = getInteriorPoints(l_u_l[0].tolist() + l_u_r[0].tolist() + l_l[0].tolist(),l_u_l[1].tolist() + l_u_r[1].tolist() + l_l[1].tolist())
x_aux.extend(lipinteriorx)
y_aux.extend(lipinteriory)

#Eyes
e_l_l = inter_plot(eye_point_down_x[:],eye_point_down_y[:],'cubic')
e_u_l = inter_plot(eye_point_up_x[:],eye_point_up_y[:],'cubic')
lefteyex, lefteyey = getInteriorPoints(e_l_l[0].tolist() + e_u_l[0].tolist(), e_l_l[1].tolist() + e_u_l[1].tolist())
x_aux.extend(lefteyex)
y_aux.extend(lefteyey)

e_l_r = inter_plot(eye_point_down_x_right[:],eye_point_down_y_right[:],'cubic')
e_u_r = inter_plot(eye_point_up_x_right[:],eye_point_up_y_right[:],'cubic')
righteyex, righteyey = getInteriorPoints(e_l_r[0].tolist() + e_u_r[0].tolist(), e_l_r[1].tolist() + e_u_r[1].tolist())
x_aux.extend(righteyex)
y_aux.extend(righteyey)

temp = im[x_aux, y_aux]

val = color.rgb2lab((im[x,y]/255.).reshape(len(x),1,3)).reshape(len(x),3)
vallips = color.rgb2lab((im[x_aux,y_aux]/255.).reshape(len(x_aux),1,3)).reshape(len(x_aux),3)
# print sum(val[:,0])
L = (sum(val[:,0])-sum(vallips[:,0]))/(len(val[:,0])-len(vallips[:,0]))
A = (sum(val[:,1])-sum(vallips[:,1]))/(len(val[:,1])-len(vallips[:,1]))
bB = (sum(val[:,2])-sum(vallips[:,2]))/(len(val[:,2])-len(vallips[:,2]))

L1,A1,B1 = color.rgb2lab(np.array((R/255.,G/255.,B/255.)).reshape(1,1,3)).reshape(3,)
val[:,0] += (L1-L)*inten
val[:,1] += (A1-A)*inten
val[:,2] += (B1-bB)*inten

im[x,y] = color.lab2rgb(val.reshape(len(x),1,3)).reshape(len(x),3)*255

scale = min(width/750, height/1000)
# Blur Filter
filter = np.zeros((height,width))
cv2.fillConvexPoly(filter,np.array(c_[y, x],dtype = 'int32'),1)
# cv2.fillConvexPoly(filter,np.array(c_[yright, xright],dtype = 'int32'),1)
plt.imshow(filter)
sigma = (int(int(201 * scale)/2)*2) + 1
filter = cv2.GaussianBlur(filter,(sigma,sigma),0)

# Erosion to reduce blur size
kernel_size = int(12 * scale)
kernel = np.ones((kernel_size,kernel_size),np.uint8)
filter = cv2.erode(filter,kernel,iterations = 4)

alpha=np.zeros([height,width,3],dtype='float64')
alpha[:,:,0]=filter
alpha[:,:,1]=filter
alpha[:,:,2]=filter

immask = cv2.imread('Input.jpg')
skinalpha = checkForSkin(immask)
imshow(skinalpha*255)
gca().set_aspect('equal', adjustable='box')
show()

# xspl, yspl = getBoundaryPoints(ellipseabs , ellipseord)
# xspl, yspl = getInteriorPoints(xspl, yspl)
im = (alpha*im+(1-alpha)*im2).astype('uint8')
im = ((skinalpha)*im+(1-(skinalpha))*im2).astype('uint8')
imshow(im)
# plot(upper_ellipse[0], upper_ellipse[1], 'g-')
# plot(lower_face[0], lower_face[1], 'g-')
gca().set_aspect('equal', adjustable='box')
imsave('out1.jpg',im)
show()
