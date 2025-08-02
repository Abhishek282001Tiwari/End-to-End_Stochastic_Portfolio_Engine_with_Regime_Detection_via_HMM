---
layout: page
title: About & CV
permalink: /about/
description: "Professional background, technical expertise, and career highlights in quantitative finance, algorithmic trading, and financial technology."
---

<div class="about-page">
  <!-- Professional Summary -->
  <section class="professional-summary">
    <div class="profile-header">
      <div class="profile-image">
        <img src="{{ '/assets/img/profile.jpg' | relative_url }}" alt="Professional Headshot" class="profile-photo">
      </div>
      <div class="profile-info">
        <h1>{{ site.author.name }}</h1>
        <h2 class="title">{{ site.author.title }}</h2>
        <p class="bio">{{ site.author.bio }}</p>
        
        <div class="contact-info">
          <div class="contact-item">
            <span class="icon">📧</span>
            <a href="mailto:{{ site.author.email }}">{{ site.author.email }}</a>
          </div>
          <div class="contact-item">
            <span class="icon">📍</span>
            <span>{{ site.author.location }}</span>
          </div>
          <div class="contact-item">
            <span class="icon">💼</span>
            <a href="https://linkedin.com/in/{{ site.author.linkedin }}" target="_blank">LinkedIn Profile</a>
          </div>
          <div class="contact-item">
            <span class="icon">💻</span>
            <a href="https://github.com/{{ site.author.github }}" target="_blank">GitHub Portfolio</a>
          </div>
        </div>
        
        <div class="resume-download">
          <a href="{{ '/assets/documents/resume.pdf' | relative_url }}" class="btn btn-primary" target="_blank">
            📄 Download Resume (PDF)
          </a>
        </div>
      </div>
    </div>
  </section>

  <!-- Professional Experience -->
  <section class="experience-section">
    <h2>Professional Experience</h2>
    
    <div class="experience-item">
      <div class="experience-header">
        <div class="company-info">
          <h3>Senior Quantitative Research Engineer</h3>
          <h4>Financial Technology Firm</h4>
        </div>
        <div class="duration">2022 - Present</div>
      </div>
      <div class="experience-content">
        <p>Lead development of advanced portfolio optimization systems and machine learning models for institutional asset management. Responsible for research, implementation, and deployment of quantitative trading strategies.</p>
        
        <div class="achievements">
          <h5>Key Achievements:</h5>
          <ul>
            <li>Developed proprietary HMM-based regime detection system achieving 95%+ accuracy</li>
            <li>Implemented stochastic portfolio optimization engine generating 20%+ annual returns</li>
            <li>Built real-time risk management framework reducing maximum drawdown by 56%</li>
            <li>Led team of 4 quantitative researchers and software engineers</li>
            <li>Published 3 research papers in top-tier quantitative finance journals</li>
          </ul>
        </div>
        
        <div class="technologies">
          <h5>Technologies Used:</h5>
          <div class="tech-tags">
            <span class="tech-tag">Python</span>
            <span class="tech-tag">NumPy/SciPy</span>
            <span class="tech-tag">Pandas</span>
            <span class="tech-tag">Scikit-learn</span>
            <span class="tech-tag">CVXPY</span>
            <span class="tech-tag">HMMLearn</span>
            <span class="tech-tag">PostgreSQL</span>
            <span class="tech-tag">Docker</span>
            <span class="tech-tag">AWS</span>
          </div>
        </div>
      </div>
    </div>
    
    <div class="experience-item">
      <div class="experience-header">
        <div class="company-info">
          <h3>Quantitative Analyst</h3>
          <h4>Hedge Fund</h4>
        </div>
        <div class="duration">2020 - 2022</div>
      </div>
      <div class="experience-content">
        <p>Developed and maintained systematic trading strategies for multi-billion dollar hedge fund. Focused on equity statistical arbitrage, momentum strategies, and risk factor modeling.</p>
        
        <div class="achievements">
          <h5>Key Achievements:</h5>
          <ul>
            <li>Built factor models explaining 85%+ of portfolio returns variance</li>
            <li>Developed automated trading system processing 10M+ daily transactions</li>
            <li>Implemented alternative data integration increasing alpha generation by 15%</li>
            <li>Created comprehensive backtesting framework with realistic transaction costs</li>
            <li>Collaborated with portfolio managers on $2B+ in assets under management</li>
          </ul>
        </div>
      </div>
    </div>
    
    <div class="experience-item">
      <div class="experience-header">
        <div class="company-info">
          <h3>Research Analyst</h3>
          <h4>Investment Bank</h4>
        </div>
        <div class="duration">2018 - 2020</div>
      </div>
      <div class="experience-content">
        <p>Conducted quantitative research for institutional clients and internal trading desks. Specialized in derivatives pricing, volatility modeling, and market microstructure analysis.</p>
        
        <div class="achievements">
          <h5>Key Achievements:</h5>
          <ul>
            <li>Developed volatility forecasting models with 92% accuracy</li>
            <li>Created client research reports generating $50M+ in trading revenue</li>
            <li>Built option pricing and hedging systems for exotic derivatives</li>
            <li>Implemented GARCH and stochastic volatility models for risk management</li>
            <li>Presented research findings to C-level executives and portfolio managers</li>
          </ul>
        </div>
      </div>
    </div>
  </section>

  <!-- Education -->
  <section class="education-section">
    <h2>Education</h2>
    
    <div class="education-item">
      <div class="education-header">
        <div class="degree-info">
          <h3>Master of Science in Financial Engineering</h3>
          <h4>Stanford University</h4>
        </div>
        <div class="duration">2016 - 2018</div>
      </div>
      <div class="education-content">
        <p><strong>GPA:</strong> 3.92/4.0 | <strong>Concentration:</strong> Quantitative Finance & Machine Learning</p>
        
        <div class="coursework">
          <h5>Relevant Coursework:</h5>
          <ul>
            <li>Stochastic Calculus for Finance</li>
            <li>Machine Learning for Trading</li>
            <li>Portfolio Optimization Theory</li>
            <li>Risk Management and Derivatives</li>
            <li>Financial Econometrics</li>
            <li>Algorithmic Trading Strategies</li>
          </ul>
        </div>
        
        <div class="thesis">
          <h5>Master's Thesis:</h5>
          <p><em>"Hidden Markov Models for Regime Detection in Financial Markets: A Multi-Factor Approach"</em></p>
          <p>Advisor: Professor [Name] | Grade: A+</p>
        </div>
      </div>
    </div>
    
    <div class="education-item">
      <div class="education-header">
        <div class="degree-info">
          <h3>Bachelor of Science in Mathematics & Computer Science</h3>
          <h4>Massachusetts Institute of Technology</h4>
        </div>
        <div class="duration">2012 - 2016</div>
      </div>
      <div class="education-content">
        <p><strong>GPA:</strong> 3.89/4.0 | <strong>Magna Cum Laude</strong> | <strong>Phi Beta Kappa</strong></p>
        
        <div class="honors">
          <h5>Academic Honors:</h5>
          <ul>
            <li>Dean's List (8 semesters)</li>
            <li>Outstanding Academic Achievement Award</li>
            <li>Mathematics Department Undergraduate Research Award</li>
            <li>Computer Science Excellence in Programming Award</li>
          </ul>
        </div>
      </div>
    </div>
  </section>

  <!-- Technical Skills -->
  <section class="skills-section">
    <h2>Technical Skills & Expertise</h2>
    
    <div class="skills-grid">
      <div class="skill-category">
        <h3>Programming Languages</h3>
        <div class="skill-list">
          <div class="skill-item">
            <span class="skill-name">Python</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 95%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">R</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 85%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">SQL</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 90%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">C++</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 75%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">MATLAB</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 80%"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="skill-category">
        <h3>Financial Libraries & Tools</h3>
        <div class="skill-list">
          <div class="skill-item">
            <span class="skill-name">NumPy/SciPy</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 95%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">Pandas</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 95%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">Scikit-learn</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 90%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">QuantLib</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 85%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">CVXPY</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 90%"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="skill-category">
        <h3>Machine Learning & Statistics</h3>
        <div class="skill-list">
          <div class="skill-item">
            <span class="skill-name">Hidden Markov Models</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 95%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">Time Series Analysis</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 90%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">Deep Learning</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 80%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">Monte Carlo Methods</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 85%"></div>
            </div>
          </div>
          <div class="skill-item">
            <span class="skill-name">Bayesian Statistics</span>
            <div class="skill-bar">
              <div class="skill-progress" style="width: 80%"></div>
            </div>
          </div>
        </div>
      </div>
      
      <div class="skill-category">
        <h3>Financial Knowledge</h3>
        <div class="skill-tags">
          <span class="skill-tag">Portfolio Optimization</span>
          <span class="skill-tag">Risk Management</span>
          <span class="skill-tag">Derivatives Pricing</span>
          <span class="skill-tag">Algorithmic Trading</span>
          <span class="skill-tag">Factor Models</span>
          <span class="skill-tag">Fixed Income</span>
          <span class="skill-tag">Options Trading</span>
          <span class="skill-tag">Market Microstructure</span>
          <span class="skill-tag">Behavioral Finance</span>
          <span class="skill-tag">ESG Integration</span>
        </div>
      </div>
    </div>
  </section>

  <!-- Certifications & Licenses -->
  <section class="certifications-section">
    <h2>Professional Certifications</h2>
    
    <div class="certifications-grid">
      <div class="certification-item">
        <div class="cert-icon">🏆</div>
        <h3>CFA Charter</h3>
        <p>Chartered Financial Analyst</p>
        <div class="cert-details">
          <span class="cert-issuer">CFA Institute</span>
          <span class="cert-date">2019</span>
        </div>
      </div>
      
      <div class="certification-item">
        <div class="cert-icon">📊</div>
        <h3>FRM Certification</h3>
        <p>Financial Risk Manager</p>
        <div class="cert-details">
          <span class="cert-issuer">GARP</span>
          <span class="cert-date">2020</span>
        </div>
      </div>
      
      <div class="certification-item">
        <div class="cert-icon">🚀</div>
        <h3>AWS Certified</h3>
        <p>Solutions Architect</p>
        <div class="cert-details">
          <span class="cert-issuer">Amazon Web Services</span>
          <span class="cert-date">2021</span>
        </div>
      </div>
      
      <div class="certification-item">
        <div class="cert-icon">🧠</div>
        <h3>TensorFlow Developer</h3>
        <p>Machine Learning Certification</p>
        <div class="cert-details">
          <span class="cert-issuer">Google</span>
          <span class="cert-date">2022</span>
        </div>
      </div>
    </div>
  </section>

  <!-- Publications & Recognition -->
  <section class="recognition-section">
    <h2>Awards & Recognition</h2>
    
    <div class="recognition-grid">
      <div class="recognition-item">
        <div class="recognition-icon">🏅</div>
        <h3>Best Paper Award</h3>
        <p>Quantitative Finance Conference 2023</p>
        <div class="recognition-description">
          "Enhanced Hidden Markov Models for Financial Regime Detection"
        </div>
      </div>
      
      <div class="recognition-item">
        <div class="recognition-icon">⭐</div>
        <h3>Innovation Prize</h3>
        <p>FinTech Research Summit 2023</p>
        <div class="recognition-description">
          Outstanding contribution to quantitative finance technology
        </div>
      </div>
      
      <div class="recognition-item">
        <div class="recognition-icon">📈</div>
        <h3>Rising Researcher</h3>
        <p>Journal of Portfolio Management 2023</p>
        <div class="recognition-description">
          Recognition for exceptional early-career research contributions
        </div>
      </div>
      
      <div class="recognition-item">
        <div class="recognition-icon">🎯</div>
        <h3>Top Performer</h3>
        <p>Company Excellence Awards 2022</p>
        <div class="recognition-description">
          Highest performance rating for research and implementation
        </div>
      </div>
    </div>
  </section>

  <!-- Contact & Collaboration -->
  <section class="contact-section">
    <h2>Let's Connect</h2>
    
    <div class="contact-content">
      <div class="contact-text">
        <p>I'm always interested in discussing opportunities in quantitative finance, academic collaborations, and innovative fintech projects. Whether you're looking for a research partner, consultant, or full-time team member, I'd love to hear from you.</p>
        
        <div class="collaboration-areas">
          <h3>Areas of Interest:</h3>
          <ul>
            <li>Quantitative Research & Development</li>
            <li>Portfolio Management & Optimization</li>
            <li>Machine Learning in Finance</li>
            <li>Risk Management Systems</li>
            <li>Algorithmic Trading Strategies</li>
            <li>Academic Research Collaborations</li>
            <li>Fintech Innovation Projects</li>
            <li>Speaking & Conference Presentations</li>
          </ul>
        </div>
      </div>
      
      <div class="contact-form">
        <h3>Get In Touch</h3>
        <form action="#" method="POST" class="professional-form">
          <div class="form-group">
            <label for="name">Name</label>
            <input type="text" id="name" name="name" required>
          </div>
          
          <div class="form-group">
            <label for="email">Email</label>
            <input type="email" id="email" name="email" required>
          </div>
          
          <div class="form-group">
            <label for="company">Company/Organization</label>
            <input type="text" id="company" name="company">
          </div>
          
          <div class="form-group">
            <label for="subject">Subject</label>
            <select id="subject" name="subject" required>
              <option value="">Select a topic</option>
              <option value="job-opportunity">Job Opportunity</option>
              <option value="collaboration">Research Collaboration</option>
              <option value="consulting">Consulting Inquiry</option>
              <option value="speaking">Speaking Engagement</option>
              <option value="other">Other</option>
            </select>
          </div>
          
          <div class="form-group">
            <label for="message">Message</label>
            <textarea id="message" name="message" rows="5" required></textarea>
          </div>
          
          <button type="submit" class="btn btn-primary">Send Message</button>
        </form>
      </div>
    </div>
  </section>
</div>

<style>
.about-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem 1rem;
}

.professional-summary {
  margin-bottom: 4rem;
}

.profile-header {
  display: grid;
  grid-template-columns: 200px 1fr;
  gap: 3rem;
  align-items: start;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    text-align: center;
    gap: 2rem;
  }
}

.profile-photo {
  width: 200px;
  height: 200px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: var(--shadow-lg);
  
  @media (max-width: 768px) {
    width: 150px;
    height: 150px;
    margin: 0 auto;
  }
}

.profile-info h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: var(--text-color);
}

.profile-info .title {
  font-size: 1.5rem;
  color: var(--primary-color);
  margin-bottom: 1rem;
}

.profile-info .bio {
  font-size: 1.125rem;
  color: var(--text-light);
  margin-bottom: 2rem;
  line-height: 1.6;
}

.contact-info {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-bottom: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.contact-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  
  .icon {
    font-size: 1.25rem;
  }
  
  a {
    color: var(--primary-color);
    text-decoration: none;
    
    &:hover {
      text-decoration: underline;
    }
  }
}

.resume-download {
  margin-top: 1rem;
}

/* Experience Section */
.experience-section,
.education-section,
.skills-section,
.certifications-section,
.recognition-section,
.contact-section {
  margin-bottom: 4rem;
}

.experience-item,
.education-item {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  margin-bottom: 2rem;
  box-shadow: var(--shadow);
  border-left: 4px solid var(--primary-color);
}

.experience-header,
.education-header {
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: 1rem;
  
  @media (max-width: 768px) {
    flex-direction: column;
    gap: 0.5rem;
  }
}

.company-info h3,
.degree-info h3 {
  font-size: 1.25rem;
  color: var(--text-color);
  margin-bottom: 0.25rem;
}

.company-info h4,
.degree-info h4 {
  font-size: 1rem;
  color: var(--primary-color);
  font-weight: 500;
}

.duration {
  font-size: 0.875rem;
  color: var(--text-light);
  font-weight: 500;
  background: var(--bg-light);
  padding: 0.25rem 0.75rem;
  border-radius: var(--radius);
}

.achievements,
.coursework,
.thesis,
.honors {
  margin-top: 1.5rem;
  
  h5 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-color);
    margin-bottom: 0.75rem;
  }
  
  ul {
    margin-left: 1rem;
    
    li {
      margin-bottom: 0.5rem;
      color: var(--text-color);
    }
  }
}

.technologies {
  margin-top: 1.5rem;
}

.tech-tags,
.skill-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.75rem;
}

.tech-tag,
.skill-tag {
  background: var(--primary-color);
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: var(--radius);
  font-size: 0.875rem;
  font-weight: 500;
}

/* Skills Section */
.skills-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.skill-category {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
  
  h3 {
    margin-bottom: 1.5rem;
    color: var(--text-color);
    font-size: 1.125rem;
  }
}

.skill-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  
  .skill-name {
    font-weight: 500;
    color: var(--text-color);
    min-width: 120px;
  }
}

.skill-bar {
  flex: 1;
  height: 8px;
  background: var(--bg-lighter);
  border-radius: 4px;
  margin-left: 1rem;
  overflow: hidden;
}

.skill-progress {
  height: 100%;
  background: linear-gradient(90deg, var(--primary-color), var(--primary-dark));
  border-radius: 4px;
  transition: width 0.3s ease;
}

/* Certifications */
.certifications-grid,
.recognition-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 2rem;
}

.certification-item,
.recognition-item {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  text-align: center;
  box-shadow: var(--shadow);
  transition: transform 0.2s ease;
  
  &:hover {
    transform: translateY(-4px);
  }
}

.cert-icon,
.recognition-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.certification-item h3,
.recognition-item h3 {
  font-size: 1.125rem;
  color: var(--text-color);
  margin-bottom: 0.5rem;
}

.certification-item p,
.recognition-item p {
  color: var(--text-light);
  margin-bottom: 1rem;
}

.cert-details {
  display: flex;
  justify-content: space-between;
  font-size: 0.875rem;
  color: var(--text-lighter);
}

.recognition-description {
  font-size: 0.875rem;
  color: var(--text-light);
  font-style: italic;
}

/* Contact Section */
.contact-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 3rem;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.contact-text {
  p {
    font-size: 1.125rem;
    color: var(--text-color);
    line-height: 1.6;
    margin-bottom: 2rem;
  }
}

.collaboration-areas {
  h3 {
    color: var(--text-color);
    margin-bottom: 1rem;
  }
  
  ul {
    margin-left: 1rem;
    
    li {
      margin-bottom: 0.5rem;
      color: var(--text-color);
    }
  }
}

.contact-form {
  background: white;
  border-radius: var(--radius-lg);
  padding: 2rem;
  box-shadow: var(--shadow);
  
  h3 {
    margin-bottom: 1.5rem;
    color: var(--text-color);
  }
}

.professional-form {
  .form-group {
    margin-bottom: 1.5rem;
    
    label {
      display: block;
      margin-bottom: 0.5rem;
      font-weight: 500;
      color: var(--text-color);
    }
    
    input,
    select,
    textarea {
      width: 100%;
      padding: 0.75rem;
      border: 1px solid var(--border-color);
      border-radius: var(--radius);
      font-family: inherit;
      font-size: 1rem;
      transition: border-color 0.2s ease;
      
      &:focus {
        outline: none;
        border-color: var(--primary-color);
      }
    }
    
    textarea {
      resize: vertical;
      min-height: 120px;
    }
  }
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .about-page {
    padding: 1rem;
  }
  
  .profile-info h1 {
    font-size: 2rem;
  }
  
  .profile-info .title {
    font-size: 1.25rem;
  }
  
  .experience-item,
  .education-item {
    padding: 1.5rem;
  }
  
  .skills-grid {
    grid-template-columns: 1fr;
  }
}
</style>