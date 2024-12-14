from transformers import PreTrainedTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os

class UnimolTokenizer(PreTrainedTokenizer):
    def __init__(
            self,
            vocab_file,
            cls_token='[CLS]',
            sep_token='[SEP]', 
            pad_token='[PAD]',
            unk_token='[UNK]',
            model_max_length=512,
            **kwargs,
        ):
        
        self.vocab_file = vocab_file
        self.token2id = {}
        self.id2token = {}

        super().__init__(
            model_max_length=model_max_length,
            **kwargs,
        )

        # Special tokens
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Map special tokens to their IDs
        self.load_vocab()
        self.cls_token_id = self.token2id.get(self.cls_token)
        self.sep_token_id = self.token2id.get(self.sep_token)
        self.pad_token_id = self.token2id.get(self.pad_token)
        self.unk_token_id = self.token2id.get(self.unk_token)

        self.add_special_tokens({
            'cls_token': self.cls_token,
            'sep_token': self.sep_token,
            'pad_token': self.pad_token,
            'unk_token': self.unk_token,
        })

    def load_vocab(self):
        """Load vocabulary from the vocab file."""
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                token = line.strip()
                self.token2id[token] = idx
                self.id2token[idx] = token

    def _tokenize(self, text):
        """Tokenize a SMILES string into atom symbols."""
        mol = self._to_mol(text)
        if mol is None:
            return [self.unk_token]
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        return atoms

    def _convert_token_to_id(self, token):
        """Convert a token to its corresponding ID."""
        return self.token2id.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index):
        """Convert an ID to its corresponding token."""
        return self.id2token.get(index, self.unk_token)

    def get_distance_matrix(self, mol: Chem.Mol, use_3d=False) -> np.ndarray:
        """Generate a distance matrix from 2D or 3D coordinates."""
        if mol is None:
            return None
        if use_3d:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            coords = np.array([list(conf.GetAtomPosition(i)) for i in range(num_atoms)])
        else:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            coords = np.array([list(conf.GetAtomPosition(i))[:2] for i in range(num_atoms)])
        coords = np.pad(coords, ((1, 1), (0, 0)), mode='constant', constant_values=0)
        dist_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        return dist_mat
    
    def _to_mol(self, text):
        """Convert a SMILES string to an RDKit molecule."""
        mol = Chem.MolFromSmiles(text)
        mol = Chem.AddHs(mol)
        return mol

    def encode(self, text, add_special_tokens=True, use_3d=True):
        """Encode a SMILES string into token IDs and generate the distance matrix."""
        tokens = self._tokenize(text)
        mol = self._to_mol(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        if add_special_tokens:
            token_ids = [self.cls_token_id] + token_ids + [self.sep_token_id]
        token_ids = np.array(token_ids, dtype=np.int64)
        dist_mat = self.get_distance_matrix(mol, use_3d=use_3d).astype(np.float32)
        edge_ids = token_ids.reshape(-1, 1) * self.vocab_size + token_ids.reshape(1, -1)
        return {'input_ids': token_ids, 'dist_mat': dist_mat, 'edge_ids': edge_ids}
    
    @property
    def vocab_size(self):
        return len(self.token2id)
    
    def get_vocab(self):
        """Return the vocabulary dictionary token2id."""
        return self.token2id

    def save_vocabulary(self, save_directory, filename_prefix=""):
        """Save the tokenizer vocabulary and configuration."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        vocab_file = os.path.join(save_directory, 'vocab.txt') # disable `filename_prefix`
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for idx in range(len(self.id2token)):
                token = self.id2token[idx]
                f.write(token + '\n')
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a tokenizer from a pretrained model or path."""
        vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.txt')
        tokenizer = cls(vocab_file, **kwargs)
        return tokenizer    