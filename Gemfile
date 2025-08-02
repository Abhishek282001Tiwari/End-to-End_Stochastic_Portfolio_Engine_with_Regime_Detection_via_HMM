source "https://rubygems.org"

# Jekyll
gem "jekyll", "~> 4.3.0"

# GitHub Pages
gem "github-pages", group: :jekyll_plugins

# Plugins
gem "jekyll-feed", "~> 0.12"
gem "jekyll-sitemap"
gem "jekyll-seo-tag"

# Theme
gem "minima", "~> 2.5"

# Windows and JRuby compatibility
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories
gem "wdm", "~> 0.1.1", :platforms => [:mingw, :x64_mingw, :mswin]

# Lock `http_parser.rb` gem to `v0.6.x` on JRuby builds
gem "http_parser.rb", "~> 0.6.0", :platforms => [:jruby]