import cv2
import numpy as np
# @Author Jessica Jasiel Renteria Escalera
# Deteccion de color
def figColor(imagenHSV):
  # Rojo
  rojoBajo1 = np.array([0, 100, 20], np.uint8)
  rojoAlto1 = np.array([10, 255, 255], np.uint8)
  rojoBajo2 = np.array([175, 100, 20], np.uint8)
  rojoAlto2 = np.array([180, 255, 255], np.uint8)

  #Verde
  verdeBajo = np.array([36, 100, 20], np.uint8)
  verdeAlto = np.array([70, 255, 255], np.uint8)

  #Azul
  azulBajo = np.array([100,100,20],np.uint8)
  azulAlto = np.array([125,255,255],np.uint8)
  
  # Naranja
  naranjaBajo = np.array([11, 100, 20], np.uint8)
  naranjaAlto = np.array([19, 255, 255], np.uint8)

  #Amarillo
  amarilloBajo = np.array([20, 100, 20], np.uint8)
  amarilloAlto = np.array([32, 255, 255], np.uint8)
  
  #Violeta
  violetaBajo = np.array([130, 100, 20], np.uint8)
  violetaAlto = np.array([145, 255, 255], np.uint8)

  #Rosa
  rosaBajo = np.array([146, 100, 20], np.uint8)
  rosaAlto = np.array([170, 255, 255], np.uint8)

  # Se buscan los colores en la imagen, segun los límites altos y bajos
  maskRojo1 = cv2.inRange(imagenHSV, rojoBajo1, rojoAlto1)
  maskRojo2 = cv2.inRange(imagenHSV, rojoBajo2, rojoAlto2)
  maskRojo = cv2.add(maskRojo1, maskRojo2)
  maskVerde = cv2.inRange(imagenHSV, verdeBajo, verdeAlto)
  maskAzul = cv2.inRange(imagenHSV, azulBajo, azulAlto)
  maskNaranja = cv2.inRange(imagenHSV, naranjaBajo, naranjaAlto)
  maskAmarillo = cv2.inRange(imagenHSV, amarilloBajo, amarilloAlto)
  maskVioleta = cv2.inRange(imagenHSV, violetaBajo, violetaAlto)
  maskRosa = cv2.inRange(imagenHSV, rosaBajo, rosaAlto)
  
  cntsRojo = cv2.findContours(maskRojo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]  
  cntsVerde = cv2.findContours(maskVerde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
  cntsAzul = cv2.findContours(maskAzul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]   
  cntsNaranja = cv2.findContours(maskNaranja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
  cntsAmarillo = cv2.findContours(maskAmarillo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
  cntsVioleta = cv2.findContours(maskVioleta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 
  cntsRosa = cv2.findContours(maskRosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] 

  if len(cntsRojo)>0: color = 'Rojo'
  elif len(cntsVerde)>0: color = 'Verde'
  elif len(cntsAzul)>0: color = 'Azul'
  elif len(cntsNaranja)>0: color = 'Naranja'
  elif len(cntsAmarillo)>0: color = 'Amarillo'
  elif len(cntsVioleta)>0: color = 'Violeta'
  elif len(cntsRosa)>0: color = 'Rosa'

  return color
    
# Deteccion de Figura
def figName(contorno,width,height):
  epsilon = 0.01*cv2.arcLength(contorno,True)
  approx = cv2.approxPolyDP(contorno,epsilon,True)

  if len(approx) == 3:
    namefig = 'Triangulo'

  if len(approx) == 4:
    aspect_ratio = float(width)/height
    if aspect_ratio > 0.95 and aspect_ratio < 1.05: namefig = 'Cuadrado'
    elif aspect_ratio < 0.95 or aspect_ratio > 1.05: namefig =  'Rectangulo'

  if len(approx) == 5:
    namefig = 'Pentagono'

  if len(approx) == 6:
    namefig = 'Hexagono'

  if len(approx) > 10:
    namefig = 'Circulo'

  return namefig
  

# Areas de las 3 figuras
def figArea(figura,width,height):

  if figura =='Cuadrado': area = width ** 2
  elif figura =='Triangulo': area = (width * height) / 2
  elif figura =='Circulo': area =  width**2 * 3.1416
  return area

#Para una imagen con una figura
def figColor_Name_Area(imagen):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10,150)
    #Para un mejorar la imagen binaria obtenida
    canny = cv2.dilate(canny,None,iterations=1)
    canny = cv2.erode(canny,None,iterations=1)
    cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    #De imagen BGR a HSV
    imageHSV = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        #Creamos una imagen auxiliar
        auxiliar = np.zeros(imagen.shape[:2], dtype="uint8")
        auxiliar = cv2.drawContours(auxiliar, [c], -1, 255, -1)
        maskHSV = cv2.bitwise_and(imageHSV,imageHSV, mask=auxiliar)
        name = figName(c,w,h)
        color = figColor(maskHSV)
        area = figArea(name,w,h)
        nameColor = name + ' ' + color
        cv2.putText(imagen,nameColor,(x,y-5),1,0.8,(69,192,143),1)
        print("  "+nameColor + " con un area de: "+ "{:.3f}".format(area))
        cv2.imshow('imagen',imagen)
        cv2.waitKey(0)

    return (area,name)


def figColor_Name_percentage(areaFigIni,nameFigIni,imagenGroup):

    gray = cv2.cvtColor(imagenGroup, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10,150)
    #Para un mejorar la imagen binaria obtenida
    canny = cv2.dilate(canny,None,iterations=1)
    canny = cv2.erode(canny,None,iterations=1)
    cnts,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    #De imagen BGR a HSV
    imageHSV = cv2.cvtColor(imagenGroup, cv2.COLOR_BGR2HSV)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        #Creamos una imagen auxiliar
        auxiliar = np.zeros(imagenGroup.shape[:2], dtype="uint8")
        auxiliar = cv2.drawContours(auxiliar, [c], -1, 255, -1)
        maskHSV = cv2.bitwise_and(imageHSV,imageHSV, mask=auxiliar)
        name = figName(c,w,h)
        color = figColor(maskHSV)
        nameColor =  name + ' ' + color
        cv2.putText(imagenGroup,nameColor,(x,y-5),1,0.8,(0,255,0),1)
        if nameFigIni == name :
            areaFig_group = figArea(name,w,h)
            if areaFigIni == areaFig_group:
                print("  "+nameColor + " es del mismo tamanio que la figura inicial")
            else:
                resp = areaFigIni / areaFig_group
                porcentaje = (1 / resp)*100
                print("  "+nameColor+" es un "+ "{:.3f}".format(porcentaje) +" % de la figura inicial")
               

        cv2.imshow('Figuras',imagenGroup)
        cv2.waitKey(0)


# Prueba con Cuadrado
imagenIni = cv2.imread('triangulo.png')
imagenGroup = cv2.imread('figuras1.png')

print("---->  Figura de Imagen Inicial <----")
areafig_ini, namefig_ini = figColor_Name_Area(imagenIni)
print("---->  Listado de Porcentajes  <----")
figColor_Name_percentage(areafig_ini,namefig_ini,imagenGroup)

