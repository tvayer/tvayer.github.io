var idx = lunr(function () {
  this.field('title')
  this.field('excerpt')
  this.field('categories')
  this.field('tags')
  this.ref('id')

  
  
    
    
  
    
    
      this.add({
          title: "Analyse de Fourier sur le pouce",
          excerpt: "Ce post est un rapide aperçu de notions basiques sur les séries de Fourier et sur la transformée de Fourier....",
          categories: [],
          tags: [],
          id: 0
      })
      
    
      this.add({
          title: "Analyse de Fourier sur l'index",
          excerpt: "Dans le post précédent nous avons vu que la théorie des fonctions intégrables n’est pas suffisante pour définir la transformée...",
          categories: [],
          tags: [],
          id: 1
      })
      
    
  
});

console.log( jQuery.type(idx) );

var store = [
  
    
    
    
  
    
    
    
      
      {
        "title": "Analyse de Fourier sur le pouce",
        "url": "https://tvayer.github.io//fourier/",
        "excerpt": "Ce post est un rapide aperçu de notions basiques sur les séries de Fourier et sur la transformée de Fourier....",
        "teaser":
          
            null
          
      },
    
      
      {
        "title": "Analyse de Fourier sur l'index",
        "url": "https://tvayer.github.io//fourier2/",
        "excerpt": "Dans le post précédent nous avons vu que la théorie des fonctions intégrables n’est pas suffisante pour définir la transformée...",
        "teaser":
          
            null
          
      }
    
  ]

$(document).ready(function() {
  $('input#search').on('keyup', function () {
    var resultdiv = $('#results');
    var query = $(this).val().toLowerCase().replace(":", "");
    var result =
      idx.query(function (q) {
        query.split(lunr.tokenizer.separator).forEach(function (term) {
          q.term(term, {  boost: 100 })
          if(query.lastIndexOf(" ") != query.length-1){
            q.term(term, {  usePipeline: false, wildcard: lunr.Query.wildcard.TRAILING, boost: 10 })
          }
          if (term != ""){
            q.term(term, {  usePipeline: false, editDistance: 1, boost: 1 })
          }
        })
      });
    resultdiv.empty();
    resultdiv.prepend('<p class="results__found">'+result.length+' Result(s) found</p>');
    for (var item in result) {
      var ref = result[item].ref;
      if(store[ref].teaser){
        var searchitem =
          '<div class="list__item">'+
            '<article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">'+
              '<h2 class="archive__item-title" itemprop="headline">'+
                '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
              '</h2>'+
              '<div class="archive__item-teaser">'+
                '<img src="'+store[ref].teaser+'" alt="">'+
              '</div>'+
              '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt+'</p>'+
            '</article>'+
          '</div>';
      }
      else{
    	  var searchitem =
          '<div class="list__item">'+
            '<article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">'+
              '<h2 class="archive__item-title" itemprop="headline">'+
                '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
              '</h2>'+
              '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt+'</p>'+
            '</article>'+
          '</div>';
      }
      resultdiv.append(searchitem);
    }
  });
});
