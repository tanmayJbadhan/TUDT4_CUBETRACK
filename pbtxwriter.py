label_map = {
    'Artist_Info': 1,
    'Info': 2,
    'Story': 3,
    'Like/Dislike': 4,
    'Star': 5
}

def save_label_map(label_map, output_file):
    with open(output_file, 'w') as file:
        for label, id in label_map.items():
            file.write('item {\n')
            file.write('  name: "{}"\n'.format(label))
            file.write('  id: {}\n'.format(id))
            file.write('}\n\n')

# Specify the output path for the label map file
output_file = 'label_map.pbtxt'
save_label_map(label_map, output_file)
