from typing import Optional

import torch
from torch import nn
from torch.utils.data import Dataset


class AGMMTrainDataset(Dataset):
    def __init__(
        self,
        user_seq,
        img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        **kwargs
    ) -> None:
        self.user_seq = [
            [v + 1 for v in seq[:-2]] for seq in user_seq
        ]  # cos 0 is padding
        self.num_user = num_user
        self.num_item = num_item
        self.img_emb = img_emb
        self.text_emb = text_emb
        self.max_len = max_len

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        seq = self.user_seq[index]
        labels = seq[1:][-self.max_len :]
        tokens = seq[:-1][-self.max_len :]

        img_emb = []
        gen_img_emb = []
        text_emb = []
        query_emb = []

        for i in range(len(tokens)):
            gen_img_emb.append(self.img_emb[labels[i] - 1])
            img_emb.append(self.img_emb[tokens[i] - 1])
            text_emb.append(self.text_emb[tokens[i] - 1])
            query_emb.append(self.text_emb[labels[i] - 1])

        mask_len = self.max_len - len(tokens)

        # padding
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))  # padding left
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))  # padding top

        tokens = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        tokens = zero_padding1d(tokens)
        labels = zero_padding1d(labels)

        img_emb = zero_padding2d(
            torch.zeros((0, img_emb.shape[-1]))
            if len(img_emb) == 0
            else torch.stack(img_emb)
        )
        text_emb = zero_padding2d(
            torch.zeros((0, text_emb.shape[-1]))
            if len(text_emb) == 0
            else torch.stack(text_emb)
        )
        query_emb = zero_padding2d(torch.stack(query_emb))
        gen_img_emb = zero_padding2d(torch.stack(gen_img_emb))

        return (
            tokens,
            labels,
            img_emb.float(),
            gen_img_emb.float(),
            text_emb.float(),
            query_emb.float(),
        )


class AGMMValidDataset(Dataset):
    def __init__(
        self,
        user_seq,
        img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        **kwargs
    ) -> None:
        self.u2seq = [[v + 1 for v in seq[:-2]] for seq in user_seq]  # cos 0 is padding
        self.u2answer = [[v[-2] + 1] for v in user_seq]
        self.num_user = num_user
        self.num_item = num_item
        self.img_emb = img_emb
        self.text_emb = text_emb
        self.max_len = max_len

    def __len__(self):
        return self.num_user

    def __getitem__(self, index):
        seq = self.u2seq[index]
        answer = self.u2answer[index]
        seq = seq[-self.max_len :]

        img_emb = []
        gen_img_emb = self.img_emb[answer[0] - 1]
        text_emb = []
        query_emb = self.text_emb[answer[0] - 1]

        for i in range(len(seq)):
            img_emb.append(self.img_emb[seq[i] - 1])  # item ori img
            text_emb.append(self.text_emb[seq[i] - 1])  # item text emb

        mask_len = self.max_len - len(seq)
        # padding
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))  # padding left
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))  # padding top

        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.tensor(answer, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        img_emb = zero_padding2d(torch.stack(img_emb))
        text_emb = zero_padding2d(torch.stack(text_emb))

        return (
            tokens,
            labels,
            img_emb.float(),
            gen_img_emb.float(),
            text_emb.float(),
            query_emb.float(),
        )


class AGMMTestDataset(Dataset):
    def __init__(
        self,
        user_seq,
        img_emb: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor],
        num_user: int,
        num_item: int,
        max_len: int = 30,
        **kwargs
    ) -> None:
        self.user_seq = [
            [v + 1 for v in seq[:-2]] for seq in user_seq
        ]  # cos 0 is padding
        self.u2val = [[v[-2] + 1] for v in user_seq]
        self.u2answer = [[v[-1] + 1] for v in user_seq]
        self.num_user = num_user
        self.num_item = num_item
        self.img_emb = img_emb
        self.text_emb = text_emb
        self.max_len = max_len

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        seq = self.user_seq[index] + self.u2val[index]
        answer = self.u2answer[index]
        seq = seq[-self.max_len :]

        img_emb = []
        gen_img_emb = self.img_emb[answer[0] - 1]
        text_emb = []
        query_emb = self.text_emb[answer[0] - 1]

        for i in range(len(seq)):
            img_emb.append(self.img_emb[seq[i] - 1])  # item ori img
            text_emb.append(self.text_emb[seq[i] - 1])  # item text emb

        mask_len = self.max_len - len(seq)
        # padding
        zero_padding1d = nn.ZeroPad1d((mask_len, 0))  # padding left
        zero_padding2d = nn.ZeroPad2d((0, 0, mask_len, 0))  # padding top

        tokens = torch.tensor(seq, dtype=torch.long)
        labels = torch.tensor(answer, dtype=torch.long)
        tokens = zero_padding1d(tokens)
        img_emb = zero_padding2d(torch.stack(img_emb))
        text_emb = zero_padding2d(torch.stack(text_emb))

        return (
            tokens,
            labels,
            img_emb.float(),
            gen_img_emb.float(),
            text_emb.float(),
            query_emb.float(),
        )
