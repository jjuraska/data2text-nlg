import os
import unittest

from dataset_loaders.viggo import ViggoDataset
from ontology import DatasetOntologyBuilder


class TestOntology(unittest.TestCase):
    def test_ontology_export_and_import(self):
        test_ontology_path = '_ontology_test.json'
        if os.path.isfile(test_ontology_path):
            raise FileExistsError(f'File {test_ontology_path} already exists')

        ontology_builder = DatasetOntologyBuilder(ViggoDataset, preprocess_slot_names=False)
        ontology_before_export = ontology_builder.ontology
        ontology_builder.export(file_path=test_ontology_path)

        print('>> Ontology keys:')
        print(', '.join(ontology_before_export.keys()))

        ontology_builder = DatasetOntologyBuilder(ViggoDataset, load_from_file=test_ontology_path)
        ontology_after_import = ontology_builder.ontology

        os.remove(test_ontology_path)

        self.assertEqual(ontology_before_export.keys(), ontology_after_import.keys())
        for key in ontology_before_export:
            self.assertCountEqual(ontology_before_export[key], ontology_after_import[key])


if __name__ == '__main__':
    unittest.main()
