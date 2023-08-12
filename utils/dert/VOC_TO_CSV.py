import os
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def xml_to_csv(path):
    xml_list = []
    for xml_file in os.listdir(path):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(path, xml_file))
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         float(root.find('size')[1].text),
                         float(root.find('size')[0].text),
                         float(member[4][0].text),
                         float(member[4][1].text),
                         float(member[4][2].text) - float(member[4][0].text),
                         float(member[4][3].text) - float(member[4][1].text),
                         member[0].text,
                         )
                xml_list.append(value)
    return xml_list

def main():
    parser = argparse.ArgumentParser(description='VOC to CSV converter')
    parser.add_argument('annotations', type=str, help='Path to the folder containing VOC annotations')
    args = parser.parse_args()

    image_path = args.annotations
    xml_df = xml_to_csv(image_path)
    column_name = ['image_name', 'page_width', 'page_height', 'x', 'y', 'width', 'height', 'labels']
    xml_df = pd.DataFrame(xml_df, columns=column_name)
    xml_df.to_csv('voc_labels.csv', index=None)
    print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()
