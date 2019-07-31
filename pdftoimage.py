from wand.image import Image as wi
pdf = wi(filename="cs1.pdf",resolution=100)
pdfImage = pdf.convert("jpeg")
i=1
for img in pdfImage.sequence:
    page = wi(image=img)
    page,save(filename=str(i)+".jpg")
    i+=1
