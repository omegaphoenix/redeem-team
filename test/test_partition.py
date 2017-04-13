"""Test partition script."""
import os

def test_qual():
    """Makes sure the correct qual set was generated."""

    for dir_name in ['mu', 'um']:
        generated = 'data/{}/5.dta'.format(dir_name)
        qual = 'data/{}/qual.dta'.format(dir_name)
        # Use string concatenation instead of format because
        # the sed expression uses {}.
        result = os.system("cat " + generated +
                           " | sed 's/.\{2\}$//' | diff - " + qual)
        assert result == 0
