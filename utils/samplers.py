import random
from collections import defaultdict
from torch.utils.data import Sampler


class CrossLingualPairBatchSampler(Sampler):
    """
    Cross-lingual batch sampler with multiple samples per speaker per language.

    Each batch contains:
        num_speakers_per_batch speakers
        samples_per_lang_per_speaker samples from lang 0
        samples_per_lang_per_speaker samples from lang 1

    Batch size:
        batch_size = num_speakers_per_batch * 2 * samples_per_lang_per_speaker
    """

    def __init__(
        self,
        dataset,
        batch_size,
        samples_per_lang_per_speaker=1,
        drop_last=False,
        seed=1,
        num_batches=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_lang_per_speaker = samples_per_lang_per_speaker
        self.drop_last = drop_last
        self.seed = seed
        self.rng = random.Random(seed)

        denom = 2 * samples_per_lang_per_speaker
        if batch_size % denom != 0:
            raise ValueError(
                f"batch_size must be divisible by 2 * samples_per_lang_per_speaker "
                f"({denom}), got {batch_size}"
            )

        self.num_speakers_per_batch = batch_size // denom

        speaker_lang_to_indices = defaultdict(lambda: defaultdict(list))
        for idx, (label, lang_id) in enumerate(zip(dataset.labels, dataset.lang_ids)):
            speaker_lang_to_indices[int(label)][int(lang_id)].append(idx)

        self.valid_speakers = []
        self.indices_by_speaker_lang = {}

        for spk, lang_dict in speaker_lang_to_indices.items():
            if 0 in lang_dict and 1 in lang_dict:
                if len(lang_dict[0]) > 0 and len(lang_dict[1]) > 0:
                    self.valid_speakers.append(spk)
                    self.indices_by_speaker_lang[spk] = {
                        0: lang_dict[0],
                        1: lang_dict[1],
                    }

        if len(self.valid_speakers) == 0:
            raise ValueError("No bilingual speakers found.")

        if len(self.valid_speakers) < self.num_speakers_per_batch:
            raise ValueError(
                f"Need at least {self.num_speakers_per_batch} bilingual speakers, "
                f"but found {len(self.valid_speakers)}"
            )

        if num_batches is None:
            self.num_batches = max(1, len(dataset) // batch_size)
        else:
            self.num_batches = num_batches

    def _sample_k(self, items, k):
        if len(items) >= k:
            return self.rng.sample(items, k)
        # sample with replacement if not enough utterances
        return [self.rng.choice(items) for _ in range(k)]

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_speakers = self.rng.sample(
                self.valid_speakers, self.num_speakers_per_batch
            )

            batch_indices = []

            for spk in chosen_speakers:
                idx_lang0 = self._sample_k(
                    self.indices_by_speaker_lang[spk][0],
                    self.samples_per_lang_per_speaker,
                )
                idx_lang1 = self._sample_k(
                    self.indices_by_speaker_lang[spk][1],
                    self.samples_per_lang_per_speaker,
                )

                batch_indices.extend(idx_lang0)
                batch_indices.extend(idx_lang1)

            self.rng.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches