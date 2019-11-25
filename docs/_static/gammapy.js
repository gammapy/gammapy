$(document).ready(function() {

    // add info note for past releases
    var verFile = new XMLHttpRequest();
    verFile.open("GET", "https://docs.gammapy.org/stable/index.html", true);
    verFile.onreadystatechange = function() {
      if (verFile.readyState === 4) {  // makes sure the document is ready to parse.
        if (verFile.status === 200) {  // makes sure it's found the file.
          var allText = verFile.responseText;
          var match =  allText.match(/url=\.\.\/(.*)"/i);
          var version = match[1];
          var note = '<div class="admonition note"><p class="first admonition-title" style="background-color:red">Note</p>'
          note += '<p class="last">You are not reading the most up to date version of Gammapy '
          note += 'documentation.<br/>Access the <a href="https://docs.gammapy.org/'
          note += version
          note += '/">latest stable version v'
          note += version
          note += '</a> or the <a href="https://gammapy.org/news.html#releases">list of Gammapy releases</a>.</p></div>'

          var url = window.location.href
          if (url.includes("/dev/") == false && url.includes(version) == false ) {
              var divbody = document.querySelectorAll('[role="main"]');
              var divnote = document.createElement("div");
              divnote.innerHTML = note;
              divbody[0].insertBefore(divnote, divbody[0].firstChild);
          }
        }
      }
    }
    verFile.send(null);

});
