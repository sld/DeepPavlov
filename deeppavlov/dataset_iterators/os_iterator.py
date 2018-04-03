from deeppavlov.core.common.registry import register
from deeppavlov.core.data.dataset_iterator import BasicDatasetIterator


@register('opensubtitles_iterator')
class OpensubtitlesIterator(BasicDatasetIterator):
    def split(self, *args, **kwargs):
        for dt in ['train', 'valid', 'test']:
            setattr(self, dt, OpensubtitlesIterator._extract(getattr(self, dt)))

    @staticmethod
    def _extract(data):
        """ Extracts context, response from OS data

        Args:
            data: data

        Returns:
            list of (context, response)
        """
        return data
