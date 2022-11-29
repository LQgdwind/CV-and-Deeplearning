def merge(dataset, sub_dataset):
    '''
        需要合并的Attributes:
            classes (list): List of the class names sorted alphabetically.
            class_to_idx (dict): Dict with items (class_name, class_index).
            samples (list): List of (sample path, class_index) tuples
            targets (list): The class_index value for each image in the dataset
    '''
    # 合并 classes
    dataset.classes.extend(sub_dataset.classes)
    dataset.classes = sorted(list(set(dataset.classes)))
    # 合并 class_to_idx
    dataset.class_to_idx.update(sub_dataset.class_to_idx)
    # 合并 samples
    dataset.samples.extend(sub_dataset.samples)
    # 合并 targets
    dataset.targets.extend(sub_dataset.targets)

