import SimpleITK as sitk
import pandas as pd
import numpy as np
import math

def image_resample(sitk_image, new_spacing = [1.0, 1.0, 1.0], is_label = False):
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage


def generate_dataset(annotation_file,source,dest,expected_spacing=[0.65,0.65,1.0],expected_size=[164,164,64]):
    annotation = pd.read_csv(annotation_file) 
    uids = np.unique(annotation['seriesuid'])
    total = 0
    for uid in uids:
        print('INFO:  process uid   ' + uid)
        image = sitk.ReadImage(source+uid+'.mhd', imageIO="MetaImageIO")
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8)
        mask.SetDirection(image.GetDirection())
        mask.SetOrigin(image.GetOrigin())
        mask.SetSpacing(image.GetSpacing())
        # print('INFO: origin     = ' + str(image.GetOrigin()))
        # print('INFO: size       = ' + str(image.GetSize()))
        # print('INFO: spacing    = '  + str(image.GetSpacing()))
        # print('INFO: orientaion = '  + str(image.GetDirection()))        
        for index,item in annotation[annotation.seriesuid == uid].iterrows():
            orientation = image.GetDirection()
            spacing = image.GetSpacing()
            origin  = image.GetOrigin()
            t = np.array([[orientation[0]*spacing[0],orientation[3]*spacing[1],orientation[6]*spacing[2],origin[0]], 
                          [orientation[1]*spacing[0],orientation[4]*spacing[1],orientation[7]*spacing[2],origin[1]],
                          [orientation[2]*spacing[0],orientation[5]*spacing[1],orientation[8]*spacing[2],origin[2]],
                          [0,0,0,1]])
            T = np.matrix(t)
            INV_T = T.I
            p = np.array([item[1],item[2],item[3],1])
            index = INV_T.dot(p)
            ix = round(index[0,0])
            iy = round(index[0,1])
            iz = round(index[0,2])
            # print("肺结节[%d]中心坐标：(%d,%d,%d)" % (nodule, ix,iy,iz))
            # print("肺结节[%d]大小    ： %d", % (item[4]))
            px = math.ceil(item[4]/image.GetSpacing()[0]/2)
            py = math.ceil(item[4]/image.GetSpacing()[1]/2)
            pz = math.ceil(item[4]/image.GetSpacing()[2]/2)
            for i in range(-px,px+1,1):
                for j in range(-py,py+1,1):
                    for k in range(-pz,pz+1,1):
                        if math.sqrt((i*spacing[0])**2 + (j*spacing[1])**2 + (k*spacing[2])**2) < item[4]/2:
                            mask.SetPixel(ix+i,iy+j,iz+k,1)

        # print('INFO: generate image mask')

        # 重采样
        image = image_resample(image, new_spacing = expected_spacing, is_label = False)
        mask  = image_resample(mask,  new_spacing = expected_spacing, is_label = True)

        current = 0
        for index,item in annotation[annotation.seriesuid == uid].iterrows():
            total   += 1
            current += 1
            orientation = image.GetDirection()
            spacing = image.GetSpacing()
            origin  = image.GetOrigin()
            t = np.array([[orientation[0]*spacing[0],orientation[3]*spacing[1],orientation[6]*spacing[2],origin[0]], 
                          [orientation[1]*spacing[0],orientation[4]*spacing[1],orientation[7]*spacing[2],origin[1]],
                          [orientation[2]*spacing[0],orientation[5]*spacing[1],orientation[8]*spacing[2],origin[2]],
                          [0,0,0,1]])
            T = np.matrix(t)
            INV_T = T.I
            p = np.array([item[1],item[2],item[3],1])
            index = INV_T.dot(p)
            ix = round(index[0,0])
            iy = round(index[0,1])
            iz = round(index[0,2])
            print("INFO:  nodule [%d/%d]  index (%d,%d,%d)" % (current,total,ix,iy,iz))
            # Select same subregion using ExtractImageFilter
            extract = sitk.ExtractImageFilter()
            extract.SetSize(expected_size)
            new_index = [max(0,round(ix-expected_size[0]/2)), max(0,round(iy-expected_size[1]/2)), max(0,round(iz-expected_size[2]/2))]
            new_index = [min(new_index[0],image.GetSize()[0]-expected_size[0]-1), min(new_index[1],image.GetSize()[1]-expected_size[1]-1), min(new_index[2],image.GetSize()[2]-expected_size[2]-1)]
            extract.SetIndex(new_index)

            # if (ix-expected_size[0]//2) < 0 or (iy-expected_size[1]//2) < 0 or (iz - expected_size[2]//2) < 0:
            #     extract.SetIndex([max(0,round(ix-64)), max(0,round(iy-64)), max(0,round(iz-24))])
            #     print("INFO:  Not center 1")
            # elif (ix+expected_size[0]//2) >= image.GetSize()[0] or (iy-expected_size[1]//2) >= image.GetSize()[1] or (iz - expected_size[2]//2) >= image.GetSize()[2]:
            #     extract.SetIndex([min(image.GetSize()[0]-expected_size[0]-1,round(ix-64)), max(image.GetSize()[1]-expected_size[1]-1,round(iy-64)), max(image.GetSize()[2]-expected_size[2]-1,round(iz-24))])
            #     print("INFO:  Not center 2")
            # else:
            #     extract.SetIndex([round(ix-64),round(iy-64),round(iz-24)])
            extracted_image = extract.Execute(image)

            writer = sitk.ImageFileWriter()
            writer.SetFileName(dest+uid+"."+str(current)+'.img.mha')
            writer.UseCompressionOn()
            writer.Execute(extracted_image)     

            extracted_mask = extract.Execute(mask)
            writer.SetFileName(dest+uid+"."+str(current)+'.msk.mha')
            writer.UseCompressionOn()
            writer.Execute(extracted_mask)     
            # sitk.WriteImage(extracted_image, uid+'.image.mha')    
            # extracted_mask = extract.Execute(mask)
            # sitk.WriteImage(extracted_mask, uid+'.mask.mha')  

        # writer = sitk.ImageFileWriter()
        # writer.SetFileName(dest+uid+'.mha')
        # writer.UseCompressionOn()
        # writer.Execute(mask)                       
        # #sitk.WriteImage(mask, )
        # print('INFO: nodule     = ' + str(nodule))
        # print('INFO: Finished')        
        # print('--------------------------------------------------------------------------------------------'

def main():
    generate_dataset('annotations.csv','./data/','./train/')


if __name__ == "__main__":
    # execute only if run as a script
    main()