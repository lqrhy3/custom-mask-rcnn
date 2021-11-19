import pandas as pd
from pathlib import Path
from xml.etree import ElementTree
from progress.bar import IncrementalBar


def parse_xml_file(filename: Path) -> list:
    """
    Parse from xml file bbox and class names of objects
    :param filename:
    :return:
    """

    annotations = []
    tree = ElementTree.parse(filename)
    objects = tree.findall('object')

    # Load object bounding boxes into a data frame.
    for index, object in enumerate(objects):
        annotation = {}
        bbox = object.find('bndbox')

        # Make pixel indexes 0-based
        annotation['x1'] = float(bbox.find('xmin').text) - 1
        annotation['y1'] = float(bbox.find('ymin').text) - 1
        annotation['x2'] = float(bbox.find('xmax').text) - 1
        annotation['y2'] = float(bbox.find('ymax').text) - 1
        annotation['class'] = object.find('name').text.lower().strip()
        annotation['file_name'] = tree.find('filename').text

        annotations.append(annotation)

    return annotations


def create_df_annotation(directory_annotation: str) -> None:
    """
    This function parses the given directory annotation and creates a csv-file with annotation
    :param directory_annotation: path to directory annotation
    :return:
    """
    dataframe = pd.DataFrame(columns=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'])

    list_filenames = list(Path(directory_annotation).glob('*.xml'))
    bar = IncrementalBar('Countdown', max=len(list_filenames))

    for annotation_file in list_filenames:
        bar.next()

        annotation = parse_xml_file(Path(annotation_file))
        dataframe = dataframe.append(annotation, ignore_index=True)

    bar.finish()

    print(dataframe.head())
    dataframe.to_csv('annotation.csv')


if __name__ == '__main__':
    create_df_annotation(r'PASCAL VOC 2012/Annotations')