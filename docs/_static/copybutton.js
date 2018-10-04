$(document).ready(function() {
    /* Add a [>>>] button on the top-right corner of code samples to hide
     * the >>> and ... prompts and the output and thus make the code
     * copyable. */
    var div = $('.highlight-default .highlight,' +
                '.highlight-python .highlight,' +
                '.highlight-python3 .highlight'
                )
    var pre = div.find('pre');

    // get the styles from the current theme
    pre.parent().parent().css('position', 'relative');
    var hide_text = 'Hide the prompts and output';
    var show_text = 'Show the prompts and output';
    var border_width = pre.css('border-top-width');
    var border_style = pre.css('border-top-style');
    var border_color = pre.css('border-top-color');
    var button_styles = {
        'cursor':'pointer', 'position': 'absolute', 'top': '0', 'right': '0',
        'border-color': border_color, 'border-style': border_style,
        'border-width': border_width, 'color': border_color, 'text-size': '75%',
        'font-family': 'monospace', 'padding-left': '0.2em', 'padding-right': '0.2em',
        'border-radius': '0 3px 0 0'
    }

    // create and add the button to all the code blocks that contain >>>
    div.each(function(index) {
        var jthis = $(this);
        if (jthis.find('.gp').length > 0) {
            var button = $('<span class="copybutton">&gt;&gt;&gt;</span>');
            button.css(button_styles)
            button.attr('title', hide_text);
            button.data('hidden', 'false');
            jthis.prepend(button);
        }
        // tracebacks (.gt) contain bare text elements that need to be
        // wrapped in a span to work with .nextUntil() (see later)
        jthis.find('pre:has(.gt)').contents().filter(function() {
            return ((this.nodeType == 3) && (this.data.trim().length > 0));
        }).wrap('<span>');
    });

    // define the behavior of the button when it's clicked
    $('.copybutton').click(function(e){
        e.preventDefault();
        var button = $(this);
        if (button.data('hidden') === 'false') {
            // hide the code output
            button.parent().find('.go, .gp, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'hidden');
            button.css('text-decoration', 'line-through');
            button.attr('title', show_text);
            button.data('hidden', 'true');
        } else {
            // show the code output
            button.parent().find('.go, .gp, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'visible');
            button.css('text-decoration', 'none');
            button.attr('title', hide_text);
            button.data('hidden', 'false');
        }
    });

    // add info note for past releases
    var verFile = new XMLHttpRequest();
    verFile.open("GET", "http://docs.gammapy.org/stable/index.html", true);
    verFile.onreadystatechange = function() {
      if (verFile.readyState === 4) {  // makes sure the document is ready to parse.
        if (verFile.status === 200) {  // makes sure it's found the file.
          var allText = verFile.responseText;
          var match =  allText.match(/url=\.\.\/(.*)"/i);
          var version = match[1];
          var url = window.location.href
          var note = '<div class="admonition note"><p class="first admonition-title">Note</p>'
          note += '<p class="last">You are not reading the most up to date version of Gammapy '
          note += 'documentation.<br/>Access the <a href="http://docs.gammapy.org/'
          note += version
          note += '/">latest stable version v'
          note += version
          note += '.</a></p></div>'

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
