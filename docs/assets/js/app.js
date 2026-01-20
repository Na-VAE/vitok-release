// ViTok-v2 Website - Interactive Tables & Navigation

document.addEventListener('DOMContentLoaded', () => {
    setupCocoToggle();
    setupDiv8kTabs();
    setupSmoothScrolling();
    setupActiveNavHighlight();
});

// Smooth scrolling for sidebar navigation
function setupSmoothScrolling() {
    document.querySelectorAll('.sidebar-link[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    // Update URL without jumping
                    history.pushState(null, null, href);
                }
            }
        });
    });
}

// Highlight active section in sidebar as user scrolls
function setupActiveNavHighlight() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.sidebar-link[href^="#"]');

    if (!sections.length || !navLinks.length) return;

    const observerOptions = {
        rootMargin: '-20% 0px -70% 0px',
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === '#' + id) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, observerOptions);

    sections.forEach(section => observer.observe(section));
}

// COCO Resolution Toggle (256p / 512p)
function setupCocoToggle() {
    const toggleBtns = document.querySelectorAll('.resolution-toggle .toggle-btn');
    const table256 = document.getElementById('coco-256-table');
    const table512 = document.getElementById('coco-512-table');

    if (!toggleBtns.length || !table256 || !table512) return;

    toggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            toggleBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show/hide tables
            const resolution = btn.dataset.resolution;
            if (resolution === '256') {
                table256.classList.remove('hidden');
                table512.classList.add('hidden');
            } else {
                table256.classList.add('hidden');
                table512.classList.remove('hidden');
            }
        });
    });
}

// DIV8K Resolution Tabs (1024p / 2048p / 4096p / 8192p)
function setupDiv8kTabs() {
    const tabBtns = document.querySelectorAll('.resolution-tabs .tab-btn');
    const tables = {
        'div8k-1024': document.getElementById('div8k-1024-table'),
        'div8k-2048': document.getElementById('div8k-2048-table'),
        'div8k-4096': document.getElementById('div8k-4096-table'),
        'div8k-8192': document.getElementById('div8k-8192-table')
    };

    if (!tabBtns.length) return;

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update active button
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Show/hide tables
            const tabId = btn.dataset.tab;
            Object.entries(tables).forEach(([id, table]) => {
                if (table) {
                    if (id === tabId) {
                        table.classList.remove('hidden');
                    } else {
                        table.classList.add('hidden');
                    }
                }
            });
        });
    });
}
