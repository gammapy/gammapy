/*
 * linksdl.js
 * ~~~~~~~~~~~
 *
 * javascript code to enable download links of notebooks and scripts *.py
 * https://stackoverflow.com/questions/2304941
 *
 */

document.onreadystatechange = function () {
    if (document.readyState == "interactive") {
         document.getElementsByClassName("last")[0].children[0].children[0].setAttribute("target", "_blank")
         document.getElementsByClassName("last")[0].children[2].children[1].setAttribute("download", "")
         document.getElementsByClassName("last")[0].children[2].children[2].setAttribute("download", "")
     }
}
