import fetch from 'node-fetch';

const url = 'https://api.cricapi.com/v1/match_points?apikey=3d790807-a6fa-4e7e-ba99-c7e5ea573b24&offset=0&id=0b12f428-98ab-4009-831d-493d325bc555&ruleset=0';

async function display() {
  try {
    const response = await fetch(url);
    const result = await response.text();
    console.log(result);
  } catch (error) {
    console.error(error);
  }
}

display();
