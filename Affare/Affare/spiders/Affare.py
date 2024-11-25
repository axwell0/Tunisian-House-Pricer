import scrapy
from ..items import AffareItem


class AffareSpider(scrapy.Spider):
    name = 'Affare'
    start_urls = ['https://www.affare.tn/petites-annonces/tunisie/vente-maison','https://www.affare.tn/petites-annonces/tunisie/vente-appartement?o=1']
    house_page = 1
    appartement_page = 1

    def parse_posting(self, response):
        Item = AffareItem()
        Item["url"] = response.url
        Item['Type'] = response.meta.get('Type')
        Item["title"] = response.css("div.Annonce_product_info__91ryJ h1::text").get()
        self.logger.info(f'{self.name}: Scraping {response.url}...')
        Item["price"] = response.css("span.Annonce_price__tE_l1::text").get()
        Item["location"] = ''.join(response.xpath('//div[@class="Annonce_f201510__BNC4l m-t-10"]/text()').getall())
        Item["posting_date"] = response.css("div.Annonce_f201510__BNC4l::text")[4].get()

        if response.css("div.Annonce_flx785550__AnK7v").getall():
            for item in response.css("div.Annonce_flx785550__AnK7v"):
                key = item.css("div > div::text").getall()[0]
                value = ''.join(item.css("div > div::text").getall()[1:])
                Item[key] = value
        if response.css("div.Annonce_dessto__r_nAG").getall():
            description = " ".join(response.css("div.Annonce_dessto__r_nAG p::text").getall())
            Item['description'] = description.replace(u'\xa0', u' ')
        yield Item

    def parse(self, response):
        if not (response.css("div.item_empty")):
            for posting in response.css('div.AnnoncesList_product_x__S7zyQ'):
                posting_link = posting.css('a.AnnoncesList_saz__RXM7e::attr(href)').get()
                if 'appartement' in response.url:
                    meta = {'Type': 'appartement'}
                else:
                    meta = {'Type': 'villa'}
                yield scrapy.Request(f"https://www.affare.tn{posting_link}", meta=meta,callback=self.parse_posting)
            if 'appartement' in response.url:
                AffareSpider.appartement_page += 1
                yield scrapy.Request(
                    f"https://www.affare.tn/petites-annonces/tunisie/vente-appartement?o={AffareSpider.appartement_page}",
                    callback=self.parse)
            AffareSpider.house_page += 1
            yield scrapy.Request(f"https://www.affare.tn/petites-annonces/tunisie/vente-maison?o={AffareSpider.house_page}",
                                 callback=self.parse)
