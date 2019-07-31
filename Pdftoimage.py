from wand.image import Image as wi
pdf = wi(filename='sample.pdf', resolution=300)
pdfImage = pdf.convert("jpeg")
i=1
for img in pdfImage.sequence:
    page = wi(image=img)
    page.save(filename=str(i)+".jpg")
    i +=1
