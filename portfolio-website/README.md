# Portfolio Website - Stochastic Portfolio Engine

Professional Jekyll-based portfolio website showcasing the End-to-End Stochastic Portfolio Engine with Hidden Markov Model regime detection.

## 🚀 Quick Start

### Prerequisites
- Ruby 2.7 or higher
- Bundler gem
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd portfolio-website
   ```

2. **Install dependencies**
   ```bash
   bundle install
   ```

3. **Run locally**
   ```bash
   bundle exec jekyll serve
   ```

4. **View the site**
   Open http://localhost:4000 in your browser

## 📁 Site Structure

```
portfolio-website/
├── _config.yml           # Jekyll configuration
├── _layouts/              # Page templates
│   ├── default.html       # Base layout
│   ├── home.html         # Landing page layout
│   └── page.html         # Content page layout
├── _includes/             # Reusable components
│   ├── header.html       # Site navigation
│   └── footer.html       # Site footer
├── _sass/                 # SCSS stylesheets
│   ├── _base.scss        # Base styles
│   ├── _header.scss      # Navigation styles
│   ├── _home.scss        # Landing page styles
│   ├── _documentation.scss # Documentation styles
│   └── _results.scss     # Results page styles
├── assets/
│   ├── css/
│   │   └── main.scss     # Main stylesheet
│   ├── js/
│   │   ├── main.js       # Core JavaScript
│   │   └── results.js    # Interactive charts
│   └── img/              # Images and assets
├── _documentation/        # Technical documentation
├── _research/            # Research publications
├── _results/             # Performance results
├── index.md              # Homepage
├── documentation.md      # Technical docs page
├── results.md            # Results & analytics page
├── research.md           # Research publications page
├── code.md               # Code repository page
└── about.md              # About & CV page
```

## 🎨 Customization

### Site Configuration

Edit `_config.yml` to customize:

```yaml
# Site information
title: "Your Portfolio Title"
description: "Your description"
author:
  name: "Your Name"
  title: "Your Professional Title"
  bio: "Your bio"
  email: "your.email@domain.com"
  linkedin: "your-linkedin"
  github: "your-github"

# Portfolio data
portfolio:
  project_name: "Your Project Name"
  technologies: ["Python", "NumPy", "Pandas", ...]
  key_achievements:
    - "Achievement 1"
    - "Achievement 2"
```

### Styling

The site uses SCSS for styling. Key files:

- `_sass/_base.scss`: Core styles and variables
- `_sass/_home.scss`: Landing page specific styles
- `assets/css/main.scss`: Main stylesheet that imports all partials

### Content

1. **Homepage**: Edit `index.md` and `_layouts/home.html`
2. **About Page**: Update `about.md` with your information
3. **Documentation**: Add content to `documentation.md`
4. **Research**: Update `research.md` with your publications
5. **Results**: Customize `results.md` and update charts in `assets/js/results.js`

## 📊 Interactive Features

### Charts and Visualizations

The site includes interactive charts powered by Chart.js:

- Portfolio performance comparisons
- Regime detection visualizations
- Risk-return scatter plots
- Monte Carlo simulation results

To customize charts, edit `assets/js/results.js`.

### Contact Form

The contact form includes:
- Client-side validation
- Professional styling
- Accessibility features

Configure form submission in `assets/js/main.js`.

## 🔧 Development

### Local Development

```bash
# Install dependencies
bundle install

# Run with live reload
bundle exec jekyll serve --livereload

# Build for production
bundle exec jekyll build
```

### Adding New Pages

1. Create a new markdown file in the root directory
2. Add appropriate frontmatter:
   ```yaml
   ---
   layout: page
   title: "Page Title"
   permalink: /page-url/
   description: "Page description for SEO"
   ---
   ```
3. Update navigation in `_config.yml`

### Code Quality

The site follows these standards:
- Semantic HTML5
- Accessible design (WCAG 2.1 AA)
- Responsive design (mobile-first)
- Progressive enhancement
- SEO optimized

## 🚀 Deployment

### GitHub Pages

1. Push to GitHub repository
2. Enable GitHub Pages in repository settings
3. Set source to main branch
4. Update `baseurl` in `_config.yml` if needed

### Custom Domain

1. Add CNAME file with your domain
2. Update `url` in `_config.yml`
3. Configure DNS settings with your provider

### Performance Optimization

The site includes:
- Minified CSS and JavaScript
- Optimized images
- Lazy loading
- Progressive enhancement
- Service worker (optional)

## 📱 Features

### Professional Portfolio Features
- Interactive performance dashboards
- Comprehensive technical documentation
- Academic research presentation
- Code repository showcase
- Professional CV/resume
- Contact form with validation

### Technical Features
- Responsive design
- SEO optimized
- Accessibility compliant
- Fast loading
- Progressive web app ready
- Analytics integration

### Interactive Elements
- Chart.js visualizations
- Smooth scrolling navigation
- Mobile-friendly menu
- Form validation
- Loading states
- Error handling

## 🎯 SEO & Analytics

### SEO Features
- Semantic HTML structure
- Meta tags and Open Graph
- Structured data markup
- XML sitemap
- Robots.txt
- Fast loading times

### Analytics Setup

Add your Google Analytics ID to `_config.yml`:

```yaml
google_analytics: "G-XXXXXXXXXX"
```

## 🔒 Security

### Best Practices
- No sensitive data in repository
- Secure contact form handling
- Content Security Policy headers
- HTTPS enforcement
- Input validation and sanitization

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the Jekyll documentation
- Review GitHub Issues
- Contact the development team

---

Built with ❤️ using Jekyll, Chart.js, and modern web technologies.