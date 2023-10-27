# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This python script is inspired by https://gist.github.com/chsasank/7218ca16f8d022e02a9c0deb94a310fe .

Convert jupyter notebook to sphinx gallery python script.

Usage: python ipynb_to_gallery.py notebook.ipynb (Optional)script.py
"""
import json
import pypandoc as pdoc


def convert_ipynb_to_gallery(input_file_name, output_file_name=None):
    """
    Convert jupyter notebook to sphinx gallery python script.

    Parameters
    ----------
    input_file_name: str
        Path to the jupyter notebook file.
    output_file_name: str, optional
        Path to the output python file. If None, the output file name is the same as the input file name.
        Default is None.
    """
    python_file = ""

    nb_dict = json.load(open(input_file_name))
    cells = nb_dict["cells"]

    for i, cell in enumerate(cells):
        if i == 0:
            assert cell["cell_type"] == "markdown", "First cell has to be markdown"

            md_source = "".join(cell["source"])
            rst_source = pdoc.convert_text(md_source, "rst", "md").replace("``", "`")
            python_file = '"""\n' + rst_source + '\n"""'
        else:
            if cell["cell_type"] == "markdown":
                md_source = "".join(cell["source"])
                rst_source = pdoc.convert_text(md_source, "rst", "md").replace(
                    "``", "`"
                )
                commented_source = "\n".join(["# " + x for x in rst_source.split("\n")])
                python_file = (
                    python_file + "\n\n\n" + "#" * 70 + "\n" + commented_source
                )
            elif cell["cell_type"] == "code":
                source = "".join(cell["source"])
                python_file = python_file + "\n" * 2 + source

    python_file = python_file.replace("\n%", "\n# %")
    if output_file_name is None:
        output_file_name = input_file_name.replace(".ipynb", ".py")
    if not output_file_name.endswith(".py"):
        output_file_name = output_file_name + ".py"
    open(output_file_name, "w").write(python_file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2:
        convert_ipynb_to_gallery(sys.argv[-2], sys.argv[-1])
    else:
        convert_ipynb_to_gallery(sys.argv[-1])
