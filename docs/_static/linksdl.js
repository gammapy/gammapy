/*
 * linksdl.js
 * ~~~~~~~~~~~
 *
 * javascript code to enable download links of notebooks and scripts *.py
 *
 */

$(document).ready(function() {
  document.getElementsByClassName("last")[0].children[1].setAttribute("download", "")
  document.getElementsByClassName("last")[0].children[2].setAttribute("download", "")
});
