
export let language = "en";
let langData = await fetchLanguageData(language);

/****************** Localization Stuff **************************/
//Tutorial from https://medium.com/@nohanabil/building-a-multilingual-static-website-a-step-by-step-guide-7af238cc8505

document.getElementById('lang').addEventListener('click', () => {
    if (document.getElementById('lang').textContent === '🇬🇧') {
        setLanguage('en');
    } else {
        setLanguage('de');
    }
},);

// Function to change language
async function setLanguage(lang) {
    if (lang === 'de') {
        language = 'de';
        document.getElementById('lang').textContent = '🇬🇧';
    } else {
        language = 'en';
        document.getElementById('lang').textContent = '🇩🇪';
    }
    await setLanguagePreference(language);
    langData = await fetchLanguageData(language);
    updateContent(langData);
}

// Function to set the language preference  //FixMe
function setLanguagePreference(lang) {
    localStorage.setItem('language', lang);
}

// Function to fetch language data
async function fetchLanguageData(lang) {
    const response = await fetch(`languages/${lang}.json`);
    return response.json();
}

// Function to update content based on the selected language
function updateContent(langData) {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        element.innerHTML = langData[key];
    });
}

// Call updateContent() on page load            //FixMe
window.addEventListener('DOMContentLoaded', async () => {
    const userPreferredLanguage = localStorage.getItem('language') || 'en';
    setLanguage(userPreferredLanguage);
});

export function getTranslation(identifier) {
    return langData[identifier] || identifier;
}
